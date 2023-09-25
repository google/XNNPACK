// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/bf16-gemm-minmax.yaml
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


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_SHLAND, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_SHLAND, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_SHLAND, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_SHLAND, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_SHLAND, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONFMA_ZIP, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONFMA_ZIP, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONFMA_ZIP, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONFMA_ZIP, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONFMA_ZIP, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
        .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
        .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
        .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
        .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
        .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
        .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
        .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
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
            .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(5)
      .n(8)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(5)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(5)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(5)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(5)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(5)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(5)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(6)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_6X8C2__NEONBF16_BFDOT_LANE_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, n_div_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, n_div_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, n_div_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, n_div_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, n_div_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, n_div_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_1X4C8__NEONBF16_BFMLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, n_div_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_2X4C8__NEONBF16_BFMLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(3)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, n_div_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_3X4C8__NEONBF16_BFMLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(3)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, n_div_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_4X4C8__NEONBF16_BFMLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .a_stride(11)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(11)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(19)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(5)
        .n(4)
        .k(k)
        .a_stride(83)
        .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, n_gt_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, n_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, n_div_4) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, n_div_4_strided_cn) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, n_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
        }
      }
    }
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, qmin) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, qmax) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }

  TEST(BF16_GEMM_MINMAX_5X4C8__NEONBF16_BFMLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON_BF16;
    GemmMicrokernelTester()
      .mr(5)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(5)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, xnn_init_bf16_minmax_scalar_params, xnn_pack_f16_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
