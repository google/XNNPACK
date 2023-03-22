// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x16-packw.yaml
//   Generator: tools/generate-packw-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/packw.h>
#include "packw-microkernel-tester.h"


TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, n_eq_8) {
  PackWMicrokernelTester()
    .n(8)
    .k(8)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4);
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, n_div_8) {
  PackWMicrokernelTester()
    .n(16)
    .k(8)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4);
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, n_lt_8) {
  for (size_t n = 1; n < 8; n++) {
    PackWMicrokernelTester()
      .n(n)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, n_gt_8) {
  for (size_t n = 9; n < 16; n++) {
    PackWMicrokernelTester()
      .n(n)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, k_eq_8) {
  PackWMicrokernelTester()
    .n(8)
    .k(8)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4);
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, k_div_8) {
  PackWMicrokernelTester()
    .n(8)
    .k(40)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4);
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, k_lt_8) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, k_gt_8) {
  for (size_t k = 9; k < 16; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, g_eq_2) {
  PackWMicrokernelTester()
    .g(2)
    .n(8)
    .k(8)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4);
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, null_bias) {
  for (size_t n = 1; n < 16; n++) {
    PackWMicrokernelTester()
      .nullbias(true)
      .n(n)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4);
  }
}


TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, n_eq_16) {
  PackWMicrokernelTester()
    .n(16)
    .k(16)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4);
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, n_div_16) {
  PackWMicrokernelTester()
    .n(32)
    .k(16)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4);
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, n_lt_16) {
  for (size_t n = 1; n < 16; n++) {
    PackWMicrokernelTester()
      .n(n)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, n_gt_16) {
  for (size_t n = 17; n < 32; n++) {
    PackWMicrokernelTester()
      .n(n)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, k_eq_16) {
  PackWMicrokernelTester()
    .n(16)
    .k(16)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4);
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, k_div_16) {
  PackWMicrokernelTester()
    .n(16)
    .k(80)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4);
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, k_lt_16) {
  for (size_t k = 1; k < 16; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, k_gt_16) {
  for (size_t k = 17; k < 32; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, g_eq_2) {
  PackWMicrokernelTester()
    .g(2)
    .n(16)
    .k(16)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4);
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, null_bias) {
  for (size_t n = 1; n < 32; n++) {
    PackWMicrokernelTester()
      .nullbias(true)
      .n(n)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4);
  }
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, g_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .g(2)
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 16; n++) {
      PackWMicrokernelTester()
        .nullbias(true)
        .n(n)
        .k(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_PRFM_X4, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_PRFM_X4, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_PRFM_X4, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_PRFM_X4, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_PRFM_X4, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_PRFM_X4, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_PRFM_X4, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_PRFM_X4, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_PRFM_X4, g_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .g(2)
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_PRFM_X4, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 16; n++) {
      PackWMicrokernelTester()
        .nullbias(true)
        .n(n)
        .k(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(32)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(16)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 17; n < 32; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(16)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(80)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, g_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .g(2)
      .n(16)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 32; n++) {
      PackWMicrokernelTester()
        .nullbias(true)
        .n(n)
        .k(16)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_PRFM_X4, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_PRFM_X4, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(32)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_PRFM_X4, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(16)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_PRFM_X4, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 17; n < 32; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(16)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_PRFM_X4, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_PRFM_X4, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(80)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_PRFM_X4, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_PRFM_X4, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_PRFM_X4, g_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .g(2)
      .n(16)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_PRFM_X4, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 32; n++) {
      PackWMicrokernelTester()
        .nullbias(true)
        .n(n)
        .k(16)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
