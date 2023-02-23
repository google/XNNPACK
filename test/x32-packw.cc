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


TEST(X32_PACKW_GEMM_GOI_X2__SCALAR, n_eq_2) {
  PackWMicrokernelTester()
    .n(2)
    .nr(2)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR, n_div_2) {
  for (size_t n = 4; n < 20; n += 2) {
    PackWMicrokernelTester()
      .n(n)
      .nr(2)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR, n_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(2)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR, n_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(2)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR, k_eq_2) {
  PackWMicrokernelTester()
    .k(2)
    .n(2)
    .nr(2)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR, k_div_2) {
  for (size_t n = 4; n < 20; n += 2) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(2)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR, k_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(2)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR, k_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(2)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR, null_bias) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(2)
      .nullbias(true)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar);
  }
}


TEST(X32_PACKW_GEMM_GOI_X4__SCALAR, n_eq_4) {
  PackWMicrokernelTester()
    .n(4)
    .nr(4)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR, n_div_4) {
  for (size_t n = 8; n < 40; n += 4) {
    PackWMicrokernelTester()
      .n(n)
      .nr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR, n_lt_4) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR, n_gt_4) {
  for (size_t n = 5; n < 8; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR, k_eq_4) {
  PackWMicrokernelTester()
    .k(4)
    .n(4)
    .nr(4)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR, k_div_4) {
  for (size_t n = 8; n < 40; n += 4) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR, k_lt_4) {
  for (size_t n = 1; n < 4; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR, k_gt_4) {
  for (size_t n = 5; n < 8; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR, null_bias) {
  for (size_t n = 1; n < 8; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(4)
      .nullbias(true)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar);
  }
}
