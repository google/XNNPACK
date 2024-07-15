// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x16-packw.yaml
//   Generator: tools/generate-packw-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packw.h"
#include "packw-microkernel-tester.h"


TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, k_eq_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(4)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4);
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, k_div_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(20)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4);
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, n_eq_8) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, n_div_8) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, n_lt_8) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4);
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, n_gt_8) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 9; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4);
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, g_gt_1) {
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
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4);
      }
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X8__SCALAR_INT_U4, null_bias) {
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
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4);
      }
    }
  }
}


TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, k_eq_4) {
  PackWMicrokernelTester()
    .n(16)
    .k(4)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4);
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, k_div_4) {
  PackWMicrokernelTester()
    .n(16)
    .k(20)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4);
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, n_eq_16) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, n_div_16) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, n_lt_16) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4);
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, n_gt_16) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 17; n < 32; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4);
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, g_gt_1) {
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
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4);
      }
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X16__SCALAR_INT_U4, null_bias) {
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
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4);
      }
    }
  }
}


TEST(X16_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, k_eq_4) {
  PackWMicrokernelTester()
    .n(32)
    .k(4)
    .nr(32)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
}

TEST(X16_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, k_div_4) {
  PackWMicrokernelTester()
    .n(32)
    .k(20)
    .nr(32)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
}

TEST(X16_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, n_eq_32) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, n_div_32) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(64)
      .k(k)
      .nr(32)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, n_lt_32) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 32; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(32)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, n_gt_32) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 33; n < 64; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(32)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, g_gt_1) {
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
          .Test(xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
      }
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X32__SCALAR_INT_U4, null_bias) {
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
          .Test(xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
      }
    }
  }
}


TEST(X16_PACKW_GEMM_GOI_X64__SCALAR_INT_U4, k_eq_4) {
  PackWMicrokernelTester()
    .n(64)
    .k(4)
    .nr(64)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
}

TEST(X16_PACKW_GEMM_GOI_X64__SCALAR_INT_U4, k_div_4) {
  PackWMicrokernelTester()
    .n(64)
    .k(20)
    .nr(64)
    .kr(1)
    .sr(1)
    .Test(xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
}

TEST(X16_PACKW_GEMM_GOI_X64__SCALAR_INT_U4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(64)
      .k(k)
      .nr(64)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X64__SCALAR_INT_U4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(64)
      .k(k)
      .nr(64)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X64__SCALAR_INT_U4, n_eq_64) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(64)
      .k(k)
      .nr(64)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X64__SCALAR_INT_U4, n_div_64) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(128)
      .k(k)
      .nr(64)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
  }
}

TEST(X16_PACKW_GEMM_GOI_X64__SCALAR_INT_U4, n_lt_64) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 64; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(64)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X64__SCALAR_INT_U4, n_gt_64) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 65; n < 128; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(64)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X64__SCALAR_INT_U4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 65; n < 128; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(64)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
      }
    }
  }
}

TEST(X16_PACKW_GEMM_GOI_X64__SCALAR_INT_U4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 65; n < 128; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(64)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
      }
    }
  }
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4, null_bias) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12, k_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(12)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12, k_div_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(60)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12, k_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 12; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12, k_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 13; k < 24; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 24; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 24; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12_PRFM, k_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(12)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12_PRFM, k_div_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(60)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12_PRFM, k_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 12; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12_PRFM, k_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 13; k < 24; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12_PRFM, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12_PRFM, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 24; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U12_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 24; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(16)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(80)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16_PRFM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(16)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16_PRFM, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(80)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16_PRFM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16_PRFM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16_PRFM, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16_PRFM, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U16_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4, null_bias) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(40)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(40)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12, k_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(12)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12, k_div_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(60)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12, k_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 12; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12, k_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 13; k < 24; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 24; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 24; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12_PRFM, k_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(12)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12_PRFM, k_div_12) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(60)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12_PRFM, k_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 12; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12_PRFM, k_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 13; k < 24; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12_PRFM, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12_PRFM, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 24; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 24; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U12_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 24; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(80)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16_PRFM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16_PRFM, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(80)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16_PRFM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16_PRFM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16_PRFM, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16_PRFM, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U16_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16, k_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    PackWMicrokernelTester()
      .n(8)
      .k(16)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16, k_div_16) {
    TEST_REQUIRES_X86_AVX2;
    PackWMicrokernelTester()
      .n(8)
      .k(80)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16, k_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16, k_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 17; k < 32; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16, n_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16, n_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16, g_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16, null_bias) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16_PRFM, k_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    PackWMicrokernelTester()
      .n(8)
      .k(16)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16_PRFM, k_div_16) {
    TEST_REQUIRES_X86_AVX2;
    PackWMicrokernelTester()
      .n(8)
      .k(80)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16_PRFM, k_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16_PRFM, k_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 17; k < 32; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16_PRFM, n_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16_PRFM, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16_PRFM, n_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X8__AVX2_U16_PRFM, null_bias) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16, k_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    PackWMicrokernelTester()
      .n(16)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16, k_div_16) {
    TEST_REQUIRES_X86_AVX2;
    PackWMicrokernelTester()
      .n(16)
      .k(80)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16, k_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16, k_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 17; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16, n_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16, n_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16, n_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16, n_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16, g_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16, null_bias) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16_PRFM, k_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    PackWMicrokernelTester()
      .n(16)
      .k(16)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16_PRFM, k_div_16) {
    TEST_REQUIRES_X86_AVX2;
    PackWMicrokernelTester()
      .n(16)
      .k(80)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16_PRFM, k_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16_PRFM, k_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 17; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16_PRFM, n_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16_PRFM, n_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16_PRFM, n_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16_PRFM, n_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k < 32; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
        }
      }
    }
  }

  TEST(X16_PACKW_GEMM_GOI_X16__AVX2_U16_PRFM, null_bias) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 32; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
