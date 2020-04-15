// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-ppmm-minmax.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_4X8__NEON, k_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_eq_1_subtile) {
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
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEON, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
  }

  TEST(F32_PPMM_MINMAX_4X8__NEONFMA, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_8X8__NEON, k_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_eq_1_subtile) {
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
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEON, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, k_eq_1_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, k_eq_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 8; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, k_eq_1_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, k_eq_1_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, k_gt_1_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
  }

  TEST(F32_PPMM_MINMAX_8X8__NEONFMA, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PPMM_MINMAX_4X8__SSE, k_eq_1) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, strided_cn) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_eq_1_strided_a) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_eq_1_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_gt_1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, k_gt_1_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_gt_8_strided_a) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_div_8_strided_a) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, n_div_8_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, strided_cm_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, qmin) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, qmax) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
  }

  TEST(F32_PPMM_MINMAX_4X8__SSE, strided_cm) {
    TEST_REQUIRES_X86_SSE;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_PPMM_MINMAX_4X8__PSIMD, k_eq_1) {
    TEST_REQUIRES_PSIMD;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, strided_cn) {
    TEST_REQUIRES_PSIMD;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, k_eq_1_strided_a) {
    TEST_REQUIRES_PSIMD;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, k_eq_1_subtile) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, k_eq_1_subtile_m) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, k_eq_1_subtile_n) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, k_gt_1) {
    TEST_REQUIRES_PSIMD;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, k_gt_1_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t k = 2; k < 10; k++) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, n_gt_8) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, n_gt_8_strided_cn) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, n_gt_8_strided_a) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, n_gt_8_subtile) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, n_div_8) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, n_div_8_strided_cn) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, n_div_8_strided_a) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, n_div_8_subtile) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, strided_cm_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t k = 1; k <= 5; k += 2) {
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
            .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, qmin) {
    TEST_REQUIRES_PSIMD;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, qmax) {
    TEST_REQUIRES_PSIMD;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_PPMM_MINMAX_4X8__PSIMD, strided_cm) {
    TEST_REQUIRES_PSIMD;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x8__psimd, GemmMicrokernelTester::Variant::Scalar);
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .cn_stride(5)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_eq_1_subtile) {
  for (uint32_t m = 1; m <= 4; m++) {
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 2; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .cn_stride(5)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_gt_2_strided_a) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_div_2_strided_a) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 2; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .qmin(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .qmax(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_4X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .cm_stride(5)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x2__scalar, GemmMicrokernelTester::Variant::Scalar);
}


TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t m = 1; m <= 2; m++) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 2; m++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .qmin(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .qmax(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_2X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_f32_ppmm_minmax_ukernel_2x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}


TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t m = 1; m <= 4; m++) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .qmin(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .qmax(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_4X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_f32_ppmm_minmax_ukernel_4x4__scalar, GemmMicrokernelTester::Variant::Scalar);
}


TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .cn_stride(5)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_eq_1_subtile) {
  for (uint32_t m = 1; m <= 3; m++) {
    for (uint32_t n = 1; n <= 3; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 3; m++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(3)
      .kr(1)
      .sr(1)
      .m(m)
      .n(3)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 3; n++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(3)
      .kr(1)
      .sr(1)
      .m(3)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(3)
      .kr(1)
      .sr(1)
      .m(3)
      .n(3)
      .k(k)
      .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t m = 1; m <= 3; m++) {
      for (uint32_t n = 1; n <= 3; n++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(3)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_gt_3) {
  for (uint32_t n = 4; n < 6; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(3)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_gt_3_strided_cn) {
  for (uint32_t n = 4; n < 6; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(3)
        .k(k)
        .cn_stride(5)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_gt_3_strided_a) {
  for (uint32_t n = 4; n < 6; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_gt_3_subtile) {
  for (uint32_t n = 4; n < 6; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(3)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_div_3) {
  for (uint32_t n = 6; n <= 9; n += 3) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(3)
        .k(k)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_div_3_strided_cn) {
  for (uint32_t n = 6; n <= 9; n += 3) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_div_3_strided_a) {
  for (uint32_t n = 6; n <= 9; n += 3) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(3)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, n_div_3_subtile) {
  for (uint32_t n = 6; n <= 9; n += 3) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(3)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t m = 1; m <= 3; m++) {
      for (uint32_t n = 1; n <= 3; n++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(3)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .qmin(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .qmax(128)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
}

TEST(F32_PPMM_MINMAX_3X3__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(3)
    .kr(1)
    .sr(1)
    .m(3)
    .n(3)
    .k(1)
    .cm_stride(5)
    .Test(xnn_f32_ppmm_minmax_ukernel_3x3__scalar, GemmMicrokernelTester::Variant::Scalar);
}
