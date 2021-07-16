// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-igemm-minmax-gemmlowp.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
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
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
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
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
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
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
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
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
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
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, small_kernel) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .ks(3)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, small_kernel_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, n_gt_4_small_kernel) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, n_div_4_small_kernel) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
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
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, a_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, zero) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t mz = 0; mz < 2; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, qmin) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, qmax) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, no_a_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, no_b_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE2_LD64, no_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSSE3;
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
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSSE3;
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
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSSE3;
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
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSSE3;
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
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSSE3;
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
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, small_kernel) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .ks(3)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, small_kernel_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, n_gt_4_small_kernel) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, n_div_4_small_kernel) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
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
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, a_offset) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, zero) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t mz = 0; mz < 2; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, qmin) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, qmax) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, no_a_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, no_b_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSSE3_LD64, no_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
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
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
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
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
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
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
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
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
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
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, small_kernel) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .ks(3)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, small_kernel_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, n_gt_4_small_kernel) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, n_div_4_small_kernel) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 2; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
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
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, a_offset) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, zero) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t mz = 0; mz < 2; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(8)
          .sr(1)
          .m(2)
          .n(4)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, qmin) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, qmax) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(8)
      .sr(1)
      .m(2)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, no_a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, no_b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_2X4C8__SSE41_LD64, no_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(8)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_2x4c8__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, small_kernel) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .ks(3)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, small_kernel_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, n_gt_4_small_kernel) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, n_div_4_small_kernel) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, a_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, zero) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, qmin) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, qmax) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, no_a_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, no_b_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE2_LD64, no_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse2_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, small_kernel) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .ks(3)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, small_kernel_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, n_gt_4_small_kernel) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, n_div_4_small_kernel) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, a_offset) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, zero) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, qmin) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, qmax) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, no_a_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, no_b_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSSE3_LD64, no_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__ssse3_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, small_kernel) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .ks(3)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, small_kernel_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, n_gt_4_small_kernel) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, n_div_4_small_kernel) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
        }
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, a_offset) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, zero) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(4)
          .n(4)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
      }
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, qmin) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, qmax) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, no_a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, no_b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }

  TEST(QU8_IGEMM_MINMAX_GEMMLOWP_4X4C2__SSE41_LD64, no_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_igemm_minmax_gemmlowp_ukernel_4x4c2__sse41_ld64, xnn_init_qu8_conv_minmax_gemmlowp_sse2_params, xnn_init_qu8_requantization_gemmlowp_params, xnn_qu8_requantize_gemmlowp);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
