// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-ppmm-minmax.yaml
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
  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, strided_cn) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_eq_4_strided_a) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_eq_4_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_eq_4_subtile_m) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_eq_4_subtile_n) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_lt_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_lt_4_strided_a) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_lt_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_gt_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_gt_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_div_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_div_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_gt_8) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_gt_8_strided_cn) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_gt_8_strided_a) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_gt_8_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_div_8) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_div_8_strided_cn) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_div_8_strided_a) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_div_8_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, strided_cm_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, strided_cm) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cn) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_4_strided_a) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_4_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_4_subtile_m) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_4_subtile_n) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_lt_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_lt_4_strided_a) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_lt_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_gt_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_gt_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_div_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_div_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_strided_cn) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_strided_a) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_strided_cn) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_strided_a) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cm_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cm) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, strided_cn) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_strided_a) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile_m) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_eq_4_subtile_n) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4_strided_a) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_lt_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_gt_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_gt_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_div_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, k_div_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_strided_cn) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_strided_a) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_div_8) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_strided_cn) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_strided_a) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, strided_cm_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, strided_cm) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cn) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_strided_a) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_subtile_m) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_4_subtile_n) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_lt_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_lt_4_strided_a) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_lt_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_gt_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_gt_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_div_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_div_4_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_strided_cn) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_strided_a) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_strided_cn) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_strided_a) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cm_subtile) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cm) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__ASM_AARCH64_NEONFMA_LD128_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, k_eq_1_subtile) {
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
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, k_eq_1_subtile) {
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
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__AARCH64_NEONFMA_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_4X8__NEON, k_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_eq_1_subtile) {
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
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, k_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, k_eq_1_subtile) {
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
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, k_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .cn_stride(19)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .cm_stride(19)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .cn_stride(19)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__AARCH64_NEONFMA_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .cm_stride(19)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_4X16__NEON, k_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .cn_stride(19)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, k_eq_1_subtile) {
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
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, k_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .cm_stride(19)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, k_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .cn_stride(19)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, k_eq_1_subtile) {
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
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, k_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X16__NEON_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .cm_stride(19)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__AARCH64_NEONFMA_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_8X8__NEON, k_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, k_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, k_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PPMM_MINMAX_4X8__SSE, k_eq_1) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, strided_cn) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_eq_1_strided_a) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_eq_1_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_gt_1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_gt_1_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_div_8_strided_a) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_div_8_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, qmin) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, qmax) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, strided_cm) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_init_f32_minmax_sse_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, k_eq_1) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, strided_cn) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, k_eq_1_strided_a) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, k_eq_1_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, k_eq_1_subtile_m) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, k_eq_1_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, k_gt_1_subtile) {
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, n_gt_8_strided_a) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, n_div_8_strided_a) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, strided_cm_subtile) {
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, qmin) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, qmax) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, strided_cm) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, k_eq_1) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, strided_cn) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, k_eq_1_strided_a) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, k_eq_1_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, k_eq_1_subtile_m) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, k_eq_1_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, k_gt_1_subtile) {
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, n_gt_8_strided_a) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, n_div_8_strided_a) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, strided_cm_subtile) {
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, qmin) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, qmax) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }

  TEST(F32_PPMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, strided_cm) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_init_f32_minmax_wasmsimd_params, xnn_pack_f32_gemm_goi_w);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 2; m++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .qmin(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .qmax(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}


TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .cn_stride(5)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 3; n++) {
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 3; m++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(3)
      .kr(1)
      .sr(1)
      .m(m)
      .n(3)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 3; n++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(3)
      .kr(1)
      .sr(1)
      .m(3)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(3)
      .kr(1)
      .sr(1)
      .m(3)
      .n(3)
      .k(k)
      .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 3; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(3)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_gt_3) {
  for (uint32_t n = 4; n < 6; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_gt_3_strided_cn) {
  for (uint32_t n = 4; n < 6; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_gt_3_strided_a) {
  for (uint32_t n = 4; n < 6; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_gt_3_subtile) {
  for (uint32_t n = 4; n < 6; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(3)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_div_3) {
  for (uint32_t n = 6; n <= 9; n += 3) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_div_3_strided_cn) {
  for (uint32_t n = 6; n <= 9; n += 3) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_div_3_strided_a) {
  for (uint32_t n = 6; n <= 9; n += 3) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_div_3_subtile) {
  for (uint32_t n = 6; n <= 9; n += 3) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(3)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 3; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(3)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .qmin(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .qmax(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .cm_stride(5)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}


TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .cn_stride(5)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_eq_1_subtile) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_eq_1_subtile_m) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_eq_1_subtile_n) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_gt_1_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_gt_2) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_gt_2_strided_cn) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_gt_2_strided_a) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_gt_2_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_div_2) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_div_2_strided_cn) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_div_2_strided_a) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_div_2_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, strided_cm_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .qmin(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .qmax(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .cm_stride(5)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}


TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_eq_1_subtile) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_eq_1_subtile_m) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_eq_1_subtile_n) {
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
      .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_gt_1_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_gt_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_gt_4_strided_cn) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_gt_4_strided_a) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_gt_4_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_div_4) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_div_4_strided_cn) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_div_4_strided_a) {
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
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_div_4_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, strided_cm_subtile) {
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
          .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .qmin(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .qmax(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
}
