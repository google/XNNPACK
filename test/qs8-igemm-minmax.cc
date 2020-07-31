// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-igemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, small_kernel) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, small_kernel_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, n_gt_4_small_kernel) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, n_div_4_small_kernel) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, a_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, zero) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t mz = 0; mz < 1; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .ks(3)
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, qmin) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, qmax) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE2, strided_cm) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, n_gt_4) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, n_gt_4_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, n_gt_4_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, n_div_4) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, n_div_4_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, n_div_4_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, small_kernel) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, small_kernel_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, n_gt_4_small_kernel) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, n_div_4_small_kernel) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, a_offset) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, zero) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE2, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, k_eq_8) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, strided_cn) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, k_lt_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, k_gt_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, k_div_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, n_gt_4) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, n_div_4) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, small_kernel) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, small_kernel_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, n_gt_4_small_kernel) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, n_div_4_small_kernel) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, a_offset) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, zero) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t mz = 0; mz < 1; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .ks(3)
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, qmin) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, qmax) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSSE3, strided_cm) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, k_eq_8) {
    TEST_REQUIRES_X86_SSSE3;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, n_gt_4) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, n_gt_4_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, n_gt_4_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, n_div_4) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, n_div_4_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, n_div_4_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, small_kernel) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, small_kernel_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, n_gt_4_small_kernel) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, n_div_4_small_kernel) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, a_offset) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, zero) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSSE3, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cn_stride(7)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, k_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, k_eq_8_subtile_m) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(m)
        .n(4)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, k_eq_8_subtile_n) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, k_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, k_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, k_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, k_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, k_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, k_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, n_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, n_gt_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, n_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, n_div_4_strided_cn) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, small_kernel) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .ks(3)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, small_kernel_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, n_gt_4_small_kernel) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, n_div_4_small_kernel) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .ks(3)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 4; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, a_offset) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(2)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, zero) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t mz = 0; mz < 1; mz++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(2)
          .sr(1)
          .m(1)
          .n(4)
          .k(k)
          .ks(3)
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, qmin) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, qmax) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
  }

  TEST(QS8_IGEMM_MINMAX_1X4C2__SSE41, strided_cm) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(1)
      .n(4)
      .k(8)
      .cm_stride(7)
      .Test(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, k_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(2)
      .sr(1)
      .m(4)
      .n(4)
      .k(8)
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, strided_cn) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, k_eq_8_subtile) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, k_eq_8_subtile_m) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, k_eq_8_subtile_n) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, k_lt_8) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, k_lt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, k_gt_8) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, k_gt_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, k_div_8) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, k_div_8_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, n_gt_4) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, n_gt_4_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, n_gt_4_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, n_div_4) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, n_div_4_strided_cn) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, n_div_4_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, small_kernel) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, small_kernel_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, n_gt_4_small_kernel) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, n_div_4_small_kernel) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, strided_cm_subtile) {
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
            .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
        }
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, a_offset) {
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
        .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, zero) {
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
          .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
      }
    }
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, qmin) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, qmax) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
  }

  TEST(QS8_IGEMM_MINMAX_4X4C2__SSE41, strided_cm) {
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
      .Test(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
