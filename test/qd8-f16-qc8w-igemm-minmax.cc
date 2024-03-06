// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qc8w-igemm-minmax.yaml
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
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, strided_cn) {
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
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, k_eq_16_subtile) {
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, k_eq_16_subtile_m) {
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, k_eq_16_subtile_n) {
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, k_lt_16) {
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, k_lt_16_subtile) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, k_gt_16) {
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, k_gt_16_subtile) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, k_div_16) {
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, k_div_16_subtile) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, n_gt_8) {
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, n_gt_8_strided_cn) {
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, n_gt_8_subtile) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, n_div_8) {
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, n_div_8_strided_cn) {
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, n_div_8_subtile) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, small_kernel) {
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, small_kernel_subtile) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, n_gt_8_small_kernel) {
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, n_div_8_small_kernel) {
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, strided_cm_subtile) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, a_offset) {
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, zero) {
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, qmin) {
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
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, qmax) {
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
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__NEONI8MM, strided_cm) {
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
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, n_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, a_offset) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, zero) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 2; mz++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X16C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, n_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, a_offset) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, zero) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X16C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(8)
      .n(16)
      .k(16)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(8)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(8)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, k_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, n_div_16) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, a_offset) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(643)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, zero) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 8; mz++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(8)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(643)
          .zero_index(mz)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, qmin) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(8)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, qmax) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(8)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C8__NEONI8MM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_I8MM;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(8)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, k_div_4) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, a_offset) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(23)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(23)
          .zero_index(mz)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(4)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, k_div_4) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, a_offset) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
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
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_8X16C4__NEONDOTFP16ARITH, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith, xnn_init_f16_minmax_fp16arith_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, k_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, k_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, k_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, a_offset) {
    TEST_REQUIRES_X86_AVX2;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, zero) {
    TEST_REQUIRES_X86_AVX2;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_1X8C8__AVX2, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, k_eq_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, k_lt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, k_gt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, k_div_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, n_gt_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, n_div_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, a_offset) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, zero) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
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
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, qmin) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, qmax) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_2X8C8__AVXVNNI, strided_cm) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, k_eq_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, k_lt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, k_gt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, k_div_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, n_gt_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, n_div_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, a_offset) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(409)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, zero) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 5; mz++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(409)
          .zero_index(mz)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, qmin) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, qmax) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI, strided_cm) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, k_eq_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, k_lt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, k_gt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, k_div_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, n_div_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, a_offset) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, zero) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
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
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, qmin) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, qmax) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_4X8C8__AVXVNNI_PRFM, strided_cm) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, k_eq_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, k_eq_16_subtile_m) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, k_eq_16_subtile_n) {
    TEST_REQUIRES_X86_AVXVNNI;
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
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, k_lt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, k_gt_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, k_div_16) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, k_div_16_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, n_div_8) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVXVNNI;
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
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVXVNNI;
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, a_offset) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(409)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, zero) {
    TEST_REQUIRES_X86_AVXVNNI;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t mz = 0; mz < 5; mz++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(5)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(409)
          .zero_index(mz)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, qmin) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, qmax) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_5X8C8__AVXVNNI_PRFM, strided_cm) {
    TEST_REQUIRES_X86_AVXVNNI;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(5)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm, xnn_init_f16_minmax_avxvnni_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, n_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, n_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, small_kernel) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
        }
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, a_offset) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, zero) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
  }

  TEST(QD8_F16_QC8W_IGEMM_MINMAX_6X8C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avx512skx, xnn_init_f16_minmax_avx_params, xnn_pack_qs8_conv_goki_w);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
