// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-gemm-minmax-fp32.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, strided_cn) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_lt_8) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_lt_8_strided_a) {
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
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_gt_8) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_gt_8_strided_a) {
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
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_div_8) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_div_8_strided_a) {
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
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8_strided_a) {
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
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8_subtile) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8_strided_a) {
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
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8_subtile) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, qmin) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, qmax) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X8C8__AVX2, strided_cm) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, strided_cn) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_eq_8_strided_a) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_lt_8) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_lt_8_strided_a) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_gt_8) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_gt_8_strided_a) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_div_8) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_div_8_strided_a) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(8)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, n_gt_8_strided_a) {
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
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, n_gt_8_subtile) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(2)
          .n(8)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, n_div_8_strided_a) {
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
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, n_div_8_subtile) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, qmin) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, qmax) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X8C8__AVX2, strided_cm) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, strided_cn) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_eq_8_strided_a) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 3; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_lt_8) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_lt_8_strided_a) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 3; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_gt_8) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_gt_8_strided_a) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 3; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_div_8) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_div_8_strided_a) {
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
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 3; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(8)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, n_gt_8_strided_a) {
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
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, n_gt_8_subtile) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(8)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, n_div_8_strided_a) {
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
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, n_div_8_subtile) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 3; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, qmin) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, qmax) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X8C8__AVX2, strided_cm) {
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
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 1; n <= 16; n++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_1X16C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 16; n++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, n_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(16)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(16)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, n_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, n_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(16)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, n_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_2X16C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 3; m++) {
      for (uint32_t n = 1; n <= 16; n++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 3; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 3; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 3; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 3; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_3X16C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_eq_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_eq_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_eq_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 16; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_lt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_lt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_lt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_gt_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_gt_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_div_8) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_div_8_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(8)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, k_div_8_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_gt_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_div_16_strided_a) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
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
            .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GEMM_MINMAX_FP32_4X16C8__AVX512SKX, strided_cm) {
    TEST_REQUIRES_X86_AVX512SKX;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(8)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, xnn_init_qs8_conv_minmax_fp32_avx512_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
