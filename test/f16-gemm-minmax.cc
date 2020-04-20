// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(43)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 8; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(43)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .cn_stride(19)
      .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .a_stride(5)
      .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 1; n <= 16; n++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(5)
        .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(7)
        .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_div_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(23)
        .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .cm_stride(19)
      .Test(xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .cn_stride(19)
      .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .a_stride(5)
      .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 16; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(5)
        .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(7)
        .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_div_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(23)
        .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .cm_stride(19)
      .Test(xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .cn_stride(19)
      .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .a_stride(5)
      .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 16; n++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(5)
        .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(7)
        .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_div_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .a_stride(23)
        .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(16)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(16)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(16)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(13)
          .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
  }

  TEST(F16_GEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .cm_stride(19)
      .Test(xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_1X8__AARCH64_NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
        .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
          .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_4X8__AARCH64_NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(43)
        .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t m = 1; m <= 6; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_6X8__AARCH64_NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .a_stride(7)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 8; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(7)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_stride(43)
        .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t m = 1; m <= 8; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
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
            .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
        }
      }
    }
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
  }

  TEST(F16_GEMM_MINMAX_8X8__AARCH64_NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64);
  }
#endif  // XNN_ARCH_ARM64
