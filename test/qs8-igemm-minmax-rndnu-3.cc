// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-igemm-minmax-rndnu.yaml
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


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__ASM_AARCH64_NEON_MLAL_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__ASM_AARCH64_NEON_MLAL_LANE_LD64, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, k_eq_8_subtile) {
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
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, a_offset) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, zero) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, n_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, a_offset) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, zero) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c8__neoni8mm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, k_eq_16_subtile) {
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
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MLAL_LD4R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, a_offset) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, zero) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__AARCH64_NEONDOT_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C8__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X8C16__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, k_eq_8_subtile) {
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
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C2__NEON_MULL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD2R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, k_eq_8_subtile) {
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
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_1X16C4__NEON_MULL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_DUP, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C8__NEON_MULL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(2)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X8C16__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, k_eq_16_subtile) {
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
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD2R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(2)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(2)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(2)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(2)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(2)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C2S4__NEON_MULL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(2)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(2)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4__NEON_MULL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(2)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(2)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(2)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(2)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(2)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C4S2__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(2)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(16)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(16)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(16)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(16)
          .sr(1)
          .m(2)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8__NEON_MLAL_LANE_PRFM, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, k_eq_8_subtile) {
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
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2__NEON_MULL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(3)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(3)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(3)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(3)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(3)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C2S4__NEON_MULL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(3)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD1R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, k_eq_8_subtile) {
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
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4__NEON_MULL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(3)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(3)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(3)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(3)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(3)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(3)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X8C4S2__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(3)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16__NEON_MULL_ADDW_DUP, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_DUP, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD2R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(3)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(3)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(3)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C2S4__NEON_MULL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(3)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_DUP, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD2R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_LD2R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(3)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(3)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(3)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(3)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(3)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C4S2__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(3)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_3X16C8__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_3x16c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8__NEON_MULL_ADDW_DUP, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, k_eq_8_subtile) {
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
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2__NEON_MULL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, k_eq_8_subtile) {
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
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C2S4__NEON_MULL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_LD1R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD1R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(4)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(4)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(4)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C4S2__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(4)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, k_eq_8_subtile) {
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
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C8__NEON_MULL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c8__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(16)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(16)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X8C16__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(16)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x8c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, k_eq_16_subtile) {
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
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_DUP, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, k_eq_16_subtile) {
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
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_DUP, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, k_eq_16_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, k_lt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, k_lt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, k_gt_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, k_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, k_div_16) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, k_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD2R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, n_gt_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, n_gt_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, n_gt_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, n_div_16) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, n_div_16_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, n_div_16_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, n_gt_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, n_div_16_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD2R, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_eq_8_subtile) {
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
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, a_offset) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, zero) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_4X16C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, a_offset) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, zero) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 8; mz++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_IGEMM_MINMAX_RNDNU_8X16C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_igemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
