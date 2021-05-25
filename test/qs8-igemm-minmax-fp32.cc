// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-igemm-minmax-fp32.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, small_kernel) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, a_offset) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t mz = 0; mz < 1; mz++) {
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
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_1X8C8__AVX2, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_1x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, a_offset) {
    TEST_REQUIRES_X86_AVX2;
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t mz = 0; mz < 2; mz++) {
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
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_2X8C8__AVX2, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_2x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, k_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, n_gt_8) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, n_gt_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, n_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, n_div_8) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, n_div_8_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, n_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, small_kernel_subtile) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, n_gt_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, n_div_8_small_kernel) {
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
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, a_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(3)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_IGEMM_MINMAX_FP32_3X8C8__AVX2, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_fp32_ukernel_3x8c8__avx2, xnn_init_qs8_conv_minmax_fp32_avx2_params, xnn_init_qs8_requantization_fp32_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
