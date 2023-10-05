// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-gemm-goi-minmax.yaml
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
  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(43)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(43)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
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
            .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
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
            .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(43)
        .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
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
            .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_GEMM_GOI_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
