// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x32-packw.yaml
//   Generator: tools/generate-packw-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/packw.h>
#include "packw-microkernel-tester.h"


TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT, n_eq_2) {
  PackWMicrokernelTester()
    .n(2)
    .k(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT, g_eq_2) {
  PackWMicrokernelTester()
    .g(2)
    .n(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT, n_div_2) {
  PackWMicrokernelTester()
    .n(4)
    .k(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT, n_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT, n_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT, k_eq_2) {
  PackWMicrokernelTester()
    .k(2)
    .n(2)
    .nr(2)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT, k_div_2) {
  for (size_t n = 4; n < 20; n += 2) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT, k_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT, k_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT, null_bias) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .nullbias(true)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int);
  }
}


TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT, n_eq_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT, g_eq_2) {
  PackWMicrokernelTester()
    .g(2)
    .n(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT, n_div_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT, n_lt_4) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT, n_gt_4) {
  for (size_t n = 5; n < 8; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT, k_eq_4) {
  PackWMicrokernelTester()
    .k(4)
    .n(4)
    .nr(4)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT, k_div_4) {
  for (size_t n = 8; n < 40; n += 4) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT, k_lt_4) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT, k_gt_4) {
  for (size_t n = 5; n < 8; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT, null_bias) {
  for (size_t n = 1; n < 8; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .nullbias(true)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int);
  }
}


TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT, n_eq_8) {
  PackWMicrokernelTester()
    .n(8)
    .k(8)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT, g_eq_2) {
  PackWMicrokernelTester()
    .g(2)
    .n(8)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT, n_div_8) {
  PackWMicrokernelTester()
    .n(16)
    .k(8)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT, n_lt_8) {
  for (size_t n = 1; n < 8; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT, n_gt_8) {
  for (size_t n = 9; n < 16; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT, k_eq_8) {
  PackWMicrokernelTester()
    .k(8)
    .n(8)
    .nr(8)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT, k_div_8) {
  for (size_t n = 16; n < 80; n += 8) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT, k_lt_8) {
  for (size_t n = 1; n < 8; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT, k_gt_8) {
  for (size_t n = 9; n < 16; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT, null_bias) {
  for (size_t n = 1; n < 16; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .nullbias(true)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int);
  }
}


TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT, n_eq_2) {
  PackWMicrokernelTester()
    .n(2)
    .k(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT, g_eq_2) {
  PackWMicrokernelTester()
    .g(2)
    .n(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT, n_div_2) {
  PackWMicrokernelTester()
    .n(4)
    .k(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT, n_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT, n_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT, k_eq_2) {
  PackWMicrokernelTester()
    .k(2)
    .n(2)
    .nr(2)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT, k_div_2) {
  for (size_t n = 4; n < 20; n += 2) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT, k_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT, k_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT, null_bias) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(2)
      .kr(1)
      .sr(1)
      .nullbias(true)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float);
  }
}


TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT, n_eq_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT, g_eq_2) {
  PackWMicrokernelTester()
    .g(2)
    .n(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT, n_div_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT, n_lt_4) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT, n_gt_4) {
  for (size_t n = 5; n < 8; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT, k_eq_4) {
  PackWMicrokernelTester()
    .k(4)
    .n(4)
    .nr(4)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT, k_div_4) {
  for (size_t n = 8; n < 40; n += 4) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT, k_lt_4) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT, k_gt_4) {
  for (size_t n = 5; n < 8; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT, null_bias) {
  for (size_t n = 1; n < 8; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(4)
      .kr(1)
      .sr(1)
      .nullbias(true)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float);
  }
}


TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT, n_eq_8) {
  PackWMicrokernelTester()
    .n(8)
    .k(8)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT, g_eq_2) {
  PackWMicrokernelTester()
    .g(2)
    .n(8)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT, n_div_8) {
  PackWMicrokernelTester()
    .n(16)
    .k(8)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT, n_lt_8) {
  for (size_t n = 1; n < 8; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT, n_gt_8) {
  for (size_t n = 9; n < 16; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT, k_eq_8) {
  PackWMicrokernelTester()
    .k(8)
    .n(8)
    .nr(8)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT, k_div_8) {
  for (size_t n = 16; n < 80; n += 8) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT, k_lt_8) {
  for (size_t n = 1; n < 8; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT, k_gt_8) {
  for (size_t n = 9; n < 16; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT, null_bias) {
  for (size_t n = 1; n < 16; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(8)
      .kr(1)
      .sr(1)
      .nullbias(true)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float);
  }
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE, n_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(2)
      .k(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE, g_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .g(2)
      .n(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE, n_div_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(4)
      .k(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE, n_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 2; n++) {
      PackWMicrokernelTester()
        .n(n)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE, n_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 3; n < 4; n++) {
      PackWMicrokernelTester()
        .n(n)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .k(2)
      .n(2)
      .nr(2)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 20; n += 2) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 2; n++) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 3; n < 4; n++) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      PackWMicrokernelTester()
        .n(n)
        .nr(2)
        .kr(1)
        .sr(1)
        .nullbias(true)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE, g_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .g(2)
      .n(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .k(8)
      .n(8)
      .nr(8)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 16; n < 80; n += 8) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .nr(8)
        .kr(1)
        .sr(1)
        .nullbias(true)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE, n_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(12)
      .k(12)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE, g_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .g(2)
      .n(12)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE, n_div_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(24)
      .k(12)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE, n_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 12; n++) {
      PackWMicrokernelTester()
        .n(n)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE, n_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 13; n < 24; n++) {
      PackWMicrokernelTester()
        .n(n)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE, k_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .k(12)
      .n(12)
      .nr(12)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE, k_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 24; n < 120; n += 12) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE, k_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 12; n++) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE, k_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 13; n < 24; n++) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 24; n++) {
      PackWMicrokernelTester()
        .n(n)
        .nr(12)
        .kr(1)
        .sr(1)
        .nullbias(true)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
