// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-gemm-minmax-rndnu.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, strided_cn) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_eq_8_strided_a) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_eq_8_subtile) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_eq_8_subtile_m) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_eq_8_subtile_n) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_lt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_lt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_lt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_gt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_gt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_div_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_div_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, k_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, n_gt_8) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, n_gt_8_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, n_gt_8_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, n_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, n_div_8) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, n_div_8_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, n_div_8_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, n_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, strided_cm_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, qmin) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, qmax) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, strided_cm) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, no_a_zero_point) {
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
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, no_b_zero_point) {
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
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_LD64, no_zero_point) {
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
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, strided_cn) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_strided_a) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_subtile) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_subtile_m) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_subtile_n) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_lt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_lt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_lt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_gt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_gt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_div_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_div_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, k_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_gt_8) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_gt_8_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_gt_8_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_div_8) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_div_8_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_div_8_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, n_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, strided_cm_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, qmin) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, qmax) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, strided_cm) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, no_a_zero_point) {
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
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, no_b_zero_point) {
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
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__AARCH32_NEON_MLAL_LANE_PRFM_LD64, no_zero_point) {
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
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, strided_cn) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_strided_a) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_subtile) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_subtile_m) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_subtile_n) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_lt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_lt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_lt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_gt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_gt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_div_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_div_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, strided_cm_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, qmin) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, qmax) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, strided_cm) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, no_a_zero_point) {
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
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, no_b_zero_point) {
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
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, no_zero_point) {
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
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, strided_cn) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_strided_a) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_subtile) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_subtile_m) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_subtile_n) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_lt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_lt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_lt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_gt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_gt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_div_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_div_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, strided_cm_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, qmin) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, qmax) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, strided_cm) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X16__NEON_MLAL_LANE, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8C4__AARCH64_NEONDOT_CORTEX_A55, no_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8C4__NEONDOT, no_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64


#if XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X16C4__NEONDOT, no_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64


#if XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(2)
      .n(32)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(2)
      .n(32)
      .k(8)
      .cn_stride(37)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(2)
      .n(32)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 32; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(m)
        .n(32)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 32; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, n_gt_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, n_gt_32_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(37)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, n_gt_32_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, n_gt_32_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, n_div_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, n_div_32_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(37)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, n_div_32_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, n_div_32_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(37)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(2)
      .n(32)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(2)
      .n(32)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(2)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(2)
      .n(32)
      .k(8)
      .cm_stride(37)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_2X32C4__NEONDOT, no_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(2)
        .n(32)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64


#if XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X16C4__NEONDOT, no_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64


#if XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(3)
      .n(32)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(3)
      .n(32)
      .k(8)
      .cn_stride(37)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(3)
      .n(32)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 32; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(m)
        .n(32)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 32; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(3)
        .n(32)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(3)
        .n(32)
        .k(k)
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(3)
        .n(32)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(3)
        .n(32)
        .k(k)
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(3)
        .n(32)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(3)
        .n(32)
        .k(k)
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, n_gt_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, n_gt_32_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(37)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, n_gt_32_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, n_gt_32_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 33; n < 64; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, n_div_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, n_div_32_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(37)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, n_div_32_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(32)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, n_div_32_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 64; n <= 96; n += 32) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 32; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(32)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(37)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(3)
      .n(32)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(3)
      .n(32)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(3)
      .nr(32)
      .kr(4)
      .sr(1)
      .m(3)
      .n(32)
      .k(8)
      .cm_stride(37)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(3)
        .n(32)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(3)
        .n(32)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_3X32C4__NEONDOT, no_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(32)
        .kr(4)
        .sr(1)
        .m(3)
        .n(32)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64


#if XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, strided_cn) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_eq_8_subtile) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_eq_8_subtile_m) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_eq_8_subtile_n) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_lt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_lt_8_strided_a) {
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
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_lt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_gt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_gt_8_strided_a) {
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
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_div_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_div_8_strided_a) {
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
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, k_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_gt_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_gt_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_gt_16_strided_a) {
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
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_gt_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_div_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_div_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_div_16_strided_a) {
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
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, n_div_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, strided_cm_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, qmin) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, qmax) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, strided_cm) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, no_a_zero_point) {
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
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, no_b_zero_point) {
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
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16C4__NEONDOT, no_zero_point) {
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
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64


#if XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(5)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_5X16C4__NEONDOT, no_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64


#if XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_6X8C4__NEONDOT, no_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64


#if XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X8C4__NEONDOT, no_zero_point) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64


#if XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, strided_cn) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(8)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_eq_8_subtile) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_eq_8_subtile_m) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_eq_8_subtile_n) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_lt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_lt_8_strided_a) {
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
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_lt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_gt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_gt_8_strided_a) {
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
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_div_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_div_8_strided_a) {
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
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, k_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_gt_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_gt_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_gt_16_strided_a) {
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
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_gt_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_div_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_div_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_div_16_strided_a) {
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
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, n_div_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, strided_cm_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, qmin) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, qmax) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, strided_cm) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, no_a_zero_point) {
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
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, no_b_zero_point) {
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
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_8X16C4__NEONDOT, no_zero_point) {
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
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM && !XNN_PLATFORM_IOS || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, strided_cn) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_eq_8_strided_a) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_eq_8_subtile) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_eq_8_subtile_m) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_eq_8_subtile_n) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_lt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_lt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_lt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_gt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_gt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_div_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_div_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, k_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, n_gt_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, n_gt_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, n_gt_16_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, n_gt_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, n_div_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, n_div_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, n_div_16_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, n_div_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, strided_cm_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, qmin) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, qmax) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, strided_cm) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_CORTEX_A53, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, strided_cn) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_eq_8_strided_a) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_eq_8_subtile) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_eq_8_subtile_m) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_eq_8_subtile_n) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_lt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_lt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_lt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_gt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_gt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_div_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_div_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, k_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, n_gt_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, n_gt_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, n_gt_16_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, n_gt_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, n_div_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, n_div_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, n_div_16_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, n_div_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, strided_cm_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, qmin) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, qmax) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, strided_cm) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_LD64, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, strided_cn) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_eq_8_strided_a) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_eq_8_subtile) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_eq_8_subtile_m) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_eq_8_subtile_n) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_lt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_lt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_lt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_gt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_gt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_div_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_div_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, k_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, n_gt_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, n_gt_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, n_gt_16_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, n_gt_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, n_div_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, n_div_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, n_div_16_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, n_div_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, strided_cm_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, qmin) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, qmax) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, strided_cm) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_CORTEX_A75, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, strided_cn) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_strided_a) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_subtile) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_subtile_m) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_eq_8_subtile_n) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_lt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_lt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_lt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_gt_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_gt_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_gt_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_div_8) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_div_8_strided_a) {
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
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, k_div_8_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, n_gt_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, n_gt_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, n_gt_16_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, n_gt_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, n_div_16) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, n_div_16_strided_cn) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, n_div_16_strided_a) {
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
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, n_div_16_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, strided_cm_subtile) {
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
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, qmin) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, qmax) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, strided_cm) {
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
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__AARCH64_NEON_MLAL_LANE_PRFM_LD64, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
