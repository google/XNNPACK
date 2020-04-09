// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/q8-igemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(Q8_IGEMM_MINMAX_4X8__NEON, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, strided_cn) {
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
      .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, k_eq_8_subtile_m) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, k_eq_8_subtile_n) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, k_lt_8) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, k_gt_8) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, k_div_8) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, n_gt_8_subtile) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, n_div_8_strided_cn) {
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
          .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, n_div_8_subtile) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, small_kernel) {
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
        .ks(3)
        .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, a_offset) {
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
        .ks(3)
        .a_offset(163)
        .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, qmin) {
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
      .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, qmax) {
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
      .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, strided_cm) {
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
      .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, no_a_zero_point) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, no_b_zero_point) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X8__NEON, no_zero_point) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x8__neon);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(Q8_IGEMM_MINMAX_8X8__NEON, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 8; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .ks(3)
            .iterations(1)
            .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .ks(3)
          .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .ks(3)
          .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(331)
        .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 8; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(331)
          .zero_index(mz)
          .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(Q8_IGEMM_MINMAX_8X8__NEON, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_q8_igemm_minmax_ukernel_8x8__neon);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, strided_cn) {
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
      .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, k_eq_8_subtile) {
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
          .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, k_eq_8_subtile_m) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, k_eq_8_subtile_n) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, k_lt_8) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, k_lt_8_subtile) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, k_gt_8) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, k_gt_8_subtile) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, k_div_8) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, k_div_8_subtile) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, n_gt_4) {
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
          .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, n_gt_4_strided_cn) {
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
          .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, n_gt_4_subtile) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, n_div_4) {
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
          .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, n_div_4_strided_cn) {
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
          .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, n_div_4_subtile) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, small_kernel) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, small_kernel_subtile) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, n_gt_4_small_kernel) {
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
          .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, n_div_4_small_kernel) {
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
          .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, strided_cm_subtile) {
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
            .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, a_offset) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, zero) {
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
          .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, qmin) {
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
      .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, qmax) {
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
      .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, strided_cm) {
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
      .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, no_a_zero_point) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, no_b_zero_point) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(Q8_IGEMM_MINMAX_4X4C2__SSE2, no_zero_point) {
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
        .Test(xnn_q8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .cn_stride(5)
    .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, k_eq_1_subtile) {
  for (uint32_t m = 1; m <= 2; m++) {
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 2; m++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(1)
      .iterations(1)
      .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 2; n++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .cn_stride(5)
        .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, small_kernel) {
  for (size_t k = 1; k <= 5; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .ks(3)
      .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, small_kernel_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 2; n++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .ks(3)
          .iterations(1)
          .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, n_gt_2_small_kernel) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .ks(3)
        .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, n_div_2_small_kernel) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .ks(3)
        .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 2; n++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, a_offset) {
  for (size_t k = 1; k <= 5; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .ks(3)
      .a_offset(13)
      .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 2; mz++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .ks(3)
        .a_offset(13)
        .zero_index(mz)
        .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .qmin(128)
    .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .qmax(128)
    .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .cm_stride(5)
    .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, no_a_zero_point) {
  for (size_t k = 1; k <= 5; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .a_zero_point(0)
      .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, no_b_zero_point) {
  for (size_t k = 1; k <= 5; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .b_zero_point(0)
      .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_IGEMM_MINMAX_2X2__SCALAR, no_zero_point) {
  for (size_t k = 1; k <= 5; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .a_zero_point(0)
      .b_zero_point(0)
      .Test(xnn_q8_igemm_minmax_ukernel_2x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}