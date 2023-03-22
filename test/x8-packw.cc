// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x8-packw.yaml
//   Generator: tools/generate-packw-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/packw.h>
#include "packw-microkernel-tester.h"


TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_X2, n_eq_2) {
  PackWMicrokernelTester()
    .n(2)
    .k(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2);
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_X2, n_div_2) {
  PackWMicrokernelTester()
    .n(4)
    .k(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2);
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_X2, n_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    PackWMicrokernelTester()
      .n(n)
      .k(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_X2, n_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .k(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_X2, k_eq_2) {
  PackWMicrokernelTester()
    .n(2)
    .k(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2);
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_X2, k_div_2) {
  PackWMicrokernelTester()
    .n(2)
    .k(10)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2);
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_X2, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_X2, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_X2, g_eq_2) {
  PackWMicrokernelTester()
    .g(2)
    .n(2)
    .k(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2);
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_X2, null_bias) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .nullbias(true)
      .n(n)
      .k(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_x2);
  }
}


TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, n_eq_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4);
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, n_div_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4);
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, n_lt_4) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .k(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, n_gt_4) {
  for (size_t n = 5; n < 8; n++) {
    PackWMicrokernelTester()
      .n(n)
      .k(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4);
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(20)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4);
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, g_eq_2) {
  PackWMicrokernelTester()
    .g(2)
    .n(4)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4);
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, null_bias) {
  for (size_t n = 1; n < 8; n++) {
    PackWMicrokernelTester()
      .nullbias(true)
      .n(n)
      .k(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_x4);
  }
}
