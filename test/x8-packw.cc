// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x8-packw.yaml
//   Generator: tools/generate-packw-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packw.h"
#include "packw-microkernel-tester.h"


TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2, k_eq_2) {
  PackWMicrokernelTester()
    .n(2)
    .k(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2);
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2, k_div_2) {
  PackWMicrokernelTester()
    .n(2)
    .k(10)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2);
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2, n_eq_2) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2, n_div_2) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2, n_lt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 2; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2, n_gt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 3; n < 4; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2);
      }
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2);
      }
    }
  }
}


TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2, k_eq_2) {
  PackWMicrokernelTester()
    .n(4)
    .k(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2);
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2, k_div_2) {
  PackWMicrokernelTester()
    .n(4)
    .k(10)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2);
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2, n_eq_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2, n_div_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2, n_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 4; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2, n_gt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 5; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 5; n < 8; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(4)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2);
      }
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 5; n < 8; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(4)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2);
      }
    }
  }
}


TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2, k_eq_2) {
  PackWMicrokernelTester()
    .n(8)
    .k(2)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2);
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2, k_div_2) {
  PackWMicrokernelTester()
    .n(8)
    .k(10)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2);
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2, n_eq_8) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2, n_div_8) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2, n_lt_8) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2, n_gt_8) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 9; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2);
      }
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2);
      }
    }
  }
}


TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2, k_eq_2) {
  PackWMicrokernelTester()
    .n(16)
    .k(2)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2);
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2, k_div_2) {
  PackWMicrokernelTester()
    .n(16)
    .k(10)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2);
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2, n_eq_16) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2, n_div_16) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2, n_lt_16) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2, n_gt_16) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 17; n < 32; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2);
      }
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2);
      }
    }
  }
}


TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2, k_eq_2) {
  PackWMicrokernelTester()
    .n(32)
    .k(2)
    .nr(32)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2);
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2, k_div_2) {
  PackWMicrokernelTester()
    .n(32)
    .k(10)
    .nr(32)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2);
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2, n_eq_32) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2, n_div_32) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(64)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2);
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2, n_lt_32) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 32; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(32)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2, n_gt_32) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 33; n < 64; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(32)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 33; n < 64; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(32)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2);
      }
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 33; n < 64; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(32)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2);
      }
    }
  }
}


TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4, k_eq_4) {
  PackWMicrokernelTester()
    .n(2)
    .k(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4);
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4, k_div_4) {
  PackWMicrokernelTester()
    .n(2)
    .k(20)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4);
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4, n_eq_2) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4, n_div_2) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4, n_lt_2) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 2; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4, n_gt_2) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 3; n < 4; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4);
      }
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4);
      }
    }
  }
}


TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4, k_eq_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4);
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4, k_div_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(20)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4);
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4, n_eq_4) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4, n_div_4) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4, n_lt_4) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 4; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4, n_gt_4) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 5; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 5; n < 8; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(4)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4);
      }
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 5; n < 8; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(4)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4);
      }
    }
  }
}


TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, k_eq_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(4)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4);
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, k_div_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(20)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4);
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, n_eq_8) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, n_div_8) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, n_lt_8) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, n_gt_8) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 9; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4);
      }
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4);
      }
    }
  }
}


TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, k_eq_4) {
  PackWMicrokernelTester()
    .n(16)
    .k(4)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4);
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, k_div_4) {
  PackWMicrokernelTester()
    .n(16)
    .k(20)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4);
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, n_eq_16) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, n_div_16) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, n_lt_16) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, n_gt_16) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 17; n < 32; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4);
      }
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4);
      }
    }
  }
}


TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, k_eq_4) {
  PackWMicrokernelTester()
    .n(32)
    .k(4)
    .nr(32)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4);
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, k_div_4) {
  PackWMicrokernelTester()
    .n(32)
    .k(20)
    .nr(32)
    .kr(1)
    .sr(1)
    .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4);
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, n_eq_32) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, n_div_32) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(64)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4);
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, n_lt_32) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 32; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(32)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, n_gt_32) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 33; n < 64; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(32)
        .kr(1)
        .sr(1)
        .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4);
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 33; n < 64; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(32)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4);
      }
    }
  }
}

TEST(X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 33; n < 64; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(32)
          .kr(1)
          .sr(1)
          .Test(xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4);
      }
    }
  }
}
