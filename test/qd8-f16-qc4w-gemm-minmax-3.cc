// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qc4w-gemm-minmax.yaml
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


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(2)
      .n(32)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(2)
      .n(32)
      .k(16)
      .cn_stride(37)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(2)
      .n(32)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 32; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(m)
        .n(32)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 32; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, n_gt_32) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, n_gt_32_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(37)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, n_gt_32_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, n_gt_32_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, n_div_32) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, n_div_32_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(37)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, n_div_32_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, n_div_32_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(37)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(2)
      .n(32)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(2)
      .n(32)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X32C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(2)
      .n(32)
      .k(16)
      .cm_stride(37)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(4)
      .n(32)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(4)
      .n(32)
      .k(16)
      .cn_stride(37)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(4)
      .n(32)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 32; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(m)
        .n(32)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 32; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(4)
        .n(32)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(4)
        .n(32)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(4)
        .n(32)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(4)
        .n(32)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(4)
        .n(32)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(4)
        .n(32)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, n_gt_32) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, n_gt_32_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(37)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, n_gt_32_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, n_gt_32_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, n_div_32) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, n_div_32_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(37)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, n_div_32_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, n_div_32_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(37)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(4)
      .n(32)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(4)
      .n(32)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X32C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(4)
      .n(32)
      .k(16)
      .cm_stride(37)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(6)
      .n(16)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(6)
      .n(16)
      .k(16)
      .cn_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(6)
      .n(16)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(6)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, n_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(6)
      .n(16)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(6)
      .n(16)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(6)
      .n(16)
      .k(16)
      .cm_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(8)
      .n(32)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(8)
      .n(32)
      .k(16)
      .cn_stride(37)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(8)
      .n(32)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 32; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(m)
        .n(32)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 32; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(8)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(8)
        .n(32)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(8)
        .n(32)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(8)
        .n(32)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(8)
        .n(32)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(8)
        .n(32)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(32)
        .kr(8)
        .sr(1)
        .m(8)
        .n(32)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, n_gt_32) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, n_gt_32_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(37)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, n_gt_32_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, n_gt_32_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, n_div_32) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, n_div_32_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(37)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, n_div_32_strided_a) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(32)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, n_div_32_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(32)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(37)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(8)
      .n(32)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(8)
      .n(32)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X32C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(32)
      .kr(8)
      .sr(1)
      .m(8)
      .n(32)
      .k(16)
      .cm_stride(37)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cn_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .a_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cm_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .cn_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .a_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(n)
        .k(8)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .a_stride(11)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .a_stride(83)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(19)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(19)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .cm_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .a_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(n)
        .k(8)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(11)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(83)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .cn_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .a_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(6)
        .n(n)
        .k(8)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(11)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(83)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .cm_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith, xnn_init_f16_qc4w_minmax_scalar_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_eq_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_eq_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_lt_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_lt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_gt_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_div_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, n_gt_8) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, n_div_8) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, qmin) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, qmax) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX512VNNI, strided_cm) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_eq_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_eq_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_lt_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_lt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_gt_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_div_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, n_gt_8) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, n_div_8) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, qmin) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, qmax) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_5X8C8__AVX512VNNI, strided_cm) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_eq_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_eq_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_lt_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_lt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_gt_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_div_16) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, n_gt_8) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, n_div_8) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512VNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, qmin) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, qmax) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVX512VNNI, strided_cm) {
    TEST_REQUIRES_X86_AVX512VNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_eq_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(7)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(7)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_eq_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(7)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 7; m++) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t m = 1; m <= 7; m++) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(7)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_lt_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(7)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_lt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(7)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_gt_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(7)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(7)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_div_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(7)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(7)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, n_gt_8) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, n_div_8) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, qmin) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(7)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, qmax) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(7)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_7X8C8__AVX512VNNIGFNI, strided_cm) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(7)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_eq_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_eq_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_lt_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_lt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_gt_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_div_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, n_div_8) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, qmin) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, qmax) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVX512VNNIGFNI_PRFM, strided_cm) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_eq_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_eq_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_lt_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_lt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 1; k < 16; k++) {
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_gt_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 17; k < 32; k++) {
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_div_16) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 32; k <= 160; k += 16) {
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, n_div_8) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    for (size_t k = 1; k <= 80; k += 17) {
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, qmin) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, qmax) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512VNNIGFNI_PRFM, strided_cm) {
    TEST_REQUIRES_X86_AVX512VNNIGFNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_eq_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_eq_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_lt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_lt_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_gt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_div_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_div_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, n_gt_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, n_div_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, qmin) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, qmax) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI, strided_cm) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_eq_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_eq_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_lt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_lt_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_gt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_div_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_div_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, n_gt_8) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, n_div_8) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, qmin) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, qmax) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI, strided_cm) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_eq_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_eq_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(n)
        .k(16)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_lt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_lt_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_gt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_div_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_div_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, n_gt_8) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, n_div_8) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, qmin) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, qmax) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI, strided_cm) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_eq_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_eq_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .a_stride(19)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_lt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_lt_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_gt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(37)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_div_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_div_16_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(163)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, n_div_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .a_stride(83)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, qmin) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, qmax) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_1X8C8__AVXVNNI_PRFM, strided_cm) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm, xnn_init_f16_qc4w_minmax_avxvnni_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .a_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(11)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(83)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
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
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
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
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX2, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .a_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(11)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(83)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_3X8C8__AVX2, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .a_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(11)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(83)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, n_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, n_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_2X8C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .a_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
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
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(11)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(83)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, n_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, n_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
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
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
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
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_4X8C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .cn_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .a_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(n)
        .k(8)
        .iterations(1)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(11)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(19)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(83)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, n_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, n_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, n_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(43)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(8)
            .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .qmin(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .qmax(128)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }

  TEST(QD8_F16_QC4W_GEMM_MINMAX_8X8C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .cm_stride(11)
      .b_zero_point(8)
      .Test(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx, xnn_init_f16_qc4w_minmax_avx_params, xnn_pack_qs8_qc4w_gemm_goi_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
