// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x32-packx.yaml
//   Generator: tools/generate-pack-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/packx.h>
#include "pack-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKX_4X__NEON_ST4, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackMicrokernelTester()
      .mr(4)
      .m(4)
      .k(4)
      .Test(xnn_x32_packx_ukernel_4x__neon_st4);
  }

  TEST(X32_PACKX_4X__NEON_ST4, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m <= 4; m++) {
      PackMicrokernelTester()
        .mr(4)
        .m(m)
        .k(4)
        .Test(xnn_x32_packx_ukernel_4x__neon_st4);
    }
  }

  TEST(X32_PACKX_4X__NEON_ST4, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .Test(xnn_x32_packx_ukernel_4x__neon_st4);
    }
  }

  TEST(X32_PACKX_4X__NEON_ST4, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      for (size_t m = 1; m <= 4; m++) {
        PackMicrokernelTester()
          .mr(4)
          .m(m)
          .k(k)
          .Test(xnn_x32_packx_ukernel_4x__neon_st4);
      }
    }
  }

  TEST(X32_PACKX_4X__NEON_ST4, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .Test(xnn_x32_packx_ukernel_4x__neon_st4);
    }
  }

  TEST(X32_PACKX_4X__NEON_ST4, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      for (size_t m = 1; m <= 4; m++) {
        PackMicrokernelTester()
          .mr(4)
          .m(m)
          .k(k)
          .Test(xnn_x32_packx_ukernel_4x__neon_st4);
      }
    }
  }

  TEST(X32_PACKX_4X__NEON_ST4, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 8; k < 40; k += 4) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .Test(xnn_x32_packx_ukernel_4x__neon_st4);
    }
  }

  TEST(X32_PACKX_4X__NEON_ST4, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 8; k < 40; k += 4) {
      for (size_t m = 1; m <= 4; m++) {
        PackMicrokernelTester()
          .mr(4)
          .m(m)
          .k(k)
          .Test(xnn_x32_packx_ukernel_4x__neon_st4);
      }
    }
  }

  TEST(X32_PACKX_4X__NEON_ST4, strided_x) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 20; k += 5) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .x_stride(23)
        .Test(xnn_x32_packx_ukernel_4x__neon_st4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKX_4X__SSE, k_eq_4) {
    TEST_REQUIRES_X86_SSE;
    PackMicrokernelTester()
      .mr(4)
      .m(4)
      .k(4)
      .Test(xnn_x32_packx_ukernel_4x__sse);
  }

  TEST(X32_PACKX_4X__SSE, k_eq_4_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t m = 1; m <= 4; m++) {
      PackMicrokernelTester()
        .mr(4)
        .m(m)
        .k(4)
        .Test(xnn_x32_packx_ukernel_4x__sse);
    }
  }

  TEST(X32_PACKX_4X__SSE, k_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 1; k < 4; k++) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .Test(xnn_x32_packx_ukernel_4x__sse);
    }
  }

  TEST(X32_PACKX_4X__SSE, k_lt_4_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 1; k < 4; k++) {
      for (size_t m = 1; m <= 4; m++) {
        PackMicrokernelTester()
          .mr(4)
          .m(m)
          .k(k)
          .Test(xnn_x32_packx_ukernel_4x__sse);
      }
    }
  }

  TEST(X32_PACKX_4X__SSE, k_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 5; k < 8; k++) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .Test(xnn_x32_packx_ukernel_4x__sse);
    }
  }

  TEST(X32_PACKX_4X__SSE, k_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 5; k < 8; k++) {
      for (size_t m = 1; m <= 4; m++) {
        PackMicrokernelTester()
          .mr(4)
          .m(m)
          .k(k)
          .Test(xnn_x32_packx_ukernel_4x__sse);
      }
    }
  }

  TEST(X32_PACKX_4X__SSE, k_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 8; k < 40; k += 4) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .Test(xnn_x32_packx_ukernel_4x__sse);
    }
  }

  TEST(X32_PACKX_4X__SSE, k_div_4_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 8; k < 40; k += 4) {
      for (size_t m = 1; m <= 4; m++) {
        PackMicrokernelTester()
          .mr(4)
          .m(m)
          .k(k)
          .Test(xnn_x32_packx_ukernel_4x__sse);
      }
    }
  }

  TEST(X32_PACKX_4X__SSE, strided_x) {
    TEST_REQUIRES_X86_SSE;
    for (size_t k = 1; k <= 20; k += 5) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .x_stride(23)
        .Test(xnn_x32_packx_ukernel_4x__sse);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(X32_PACKX_4X__PSIMD, k_eq_4) {
    TEST_REQUIRES_PSIMD;
    PackMicrokernelTester()
      .mr(4)
      .m(4)
      .k(4)
      .Test(xnn_x32_packx_ukernel_4x__psimd);
  }

  TEST(X32_PACKX_4X__PSIMD, k_eq_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m <= 4; m++) {
      PackMicrokernelTester()
        .mr(4)
        .m(m)
        .k(4)
        .Test(xnn_x32_packx_ukernel_4x__psimd);
    }
  }

  TEST(X32_PACKX_4X__PSIMD, k_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t k = 1; k < 4; k++) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .Test(xnn_x32_packx_ukernel_4x__psimd);
    }
  }

  TEST(X32_PACKX_4X__PSIMD, k_lt_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t k = 1; k < 4; k++) {
      for (size_t m = 1; m <= 4; m++) {
        PackMicrokernelTester()
          .mr(4)
          .m(m)
          .k(k)
          .Test(xnn_x32_packx_ukernel_4x__psimd);
      }
    }
  }

  TEST(X32_PACKX_4X__PSIMD, k_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t k = 5; k < 8; k++) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .Test(xnn_x32_packx_ukernel_4x__psimd);
    }
  }

  TEST(X32_PACKX_4X__PSIMD, k_gt_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t k = 5; k < 8; k++) {
      for (size_t m = 1; m <= 4; m++) {
        PackMicrokernelTester()
          .mr(4)
          .m(m)
          .k(k)
          .Test(xnn_x32_packx_ukernel_4x__psimd);
      }
    }
  }

  TEST(X32_PACKX_4X__PSIMD, k_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t k = 8; k < 40; k += 4) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .Test(xnn_x32_packx_ukernel_4x__psimd);
    }
  }

  TEST(X32_PACKX_4X__PSIMD, k_div_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t k = 8; k < 40; k += 4) {
      for (size_t m = 1; m <= 4; m++) {
        PackMicrokernelTester()
          .mr(4)
          .m(m)
          .k(k)
          .Test(xnn_x32_packx_ukernel_4x__psimd);
      }
    }
  }

  TEST(X32_PACKX_4X__PSIMD, strided_x) {
    TEST_REQUIRES_PSIMD;
    for (size_t k = 1; k <= 20; k += 5) {
      PackMicrokernelTester()
        .mr(4)
        .m(4)
        .k(k)
        .x_stride(23)
        .Test(xnn_x32_packx_ukernel_4x__psimd);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


TEST(X32_PACKX_2X__SCALAR, k_eq_1) {
  PackMicrokernelTester()
    .mr(2)
    .m(2)
    .k(1)
    .Test(xnn_x32_packx_ukernel_2x__scalar);
}

TEST(X32_PACKX_2X__SCALAR, k_eq_1_subtile) {
  for (size_t m = 1; m <= 2; m++) {
    PackMicrokernelTester()
      .mr(2)
      .m(m)
      .k(1)
      .Test(xnn_x32_packx_ukernel_2x__scalar);
  }
}

TEST(X32_PACKX_2X__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    PackMicrokernelTester()
      .mr(2)
      .m(2)
      .k(k)
      .Test(xnn_x32_packx_ukernel_2x__scalar);
  }
}

TEST(X32_PACKX_2X__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (size_t m = 1; m <= 2; m++) {
      PackMicrokernelTester()
        .mr(2)
        .m(m)
        .k(k)
        .Test(xnn_x32_packx_ukernel_2x__scalar);
    }
  }
}

TEST(X32_PACKX_2X__SCALAR, strided_x) {
  for (size_t k = 1; k <= 5; k += 2) {
    PackMicrokernelTester()
      .mr(2)
      .m(2)
      .k(k)
      .x_stride(7)
      .Test(xnn_x32_packx_ukernel_2x__scalar);
  }
}

TEST(X32_PACKX_3X__SCALAR, k_eq_1) {
  PackMicrokernelTester()
    .mr(3)
    .m(3)
    .k(1)
    .Test(xnn_x32_packx_ukernel_3x__scalar);
}

TEST(X32_PACKX_3X__SCALAR, k_eq_1_subtile) {
  for (size_t m = 1; m <= 3; m++) {
    PackMicrokernelTester()
      .mr(3)
      .m(m)
      .k(1)
      .Test(xnn_x32_packx_ukernel_3x__scalar);
  }
}

TEST(X32_PACKX_3X__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    PackMicrokernelTester()
      .mr(3)
      .m(3)
      .k(k)
      .Test(xnn_x32_packx_ukernel_3x__scalar);
  }
}

TEST(X32_PACKX_3X__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (size_t m = 1; m <= 3; m++) {
      PackMicrokernelTester()
        .mr(3)
        .m(m)
        .k(k)
        .Test(xnn_x32_packx_ukernel_3x__scalar);
    }
  }
}

TEST(X32_PACKX_3X__SCALAR, strided_x) {
  for (size_t k = 1; k <= 5; k += 2) {
    PackMicrokernelTester()
      .mr(3)
      .m(3)
      .k(k)
      .x_stride(7)
      .Test(xnn_x32_packx_ukernel_3x__scalar);
  }
}

TEST(X32_PACKX_4X__SCALAR, k_eq_1) {
  PackMicrokernelTester()
    .mr(4)
    .m(4)
    .k(1)
    .Test(xnn_x32_packx_ukernel_4x__scalar);
}

TEST(X32_PACKX_4X__SCALAR, k_eq_1_subtile) {
  for (size_t m = 1; m <= 4; m++) {
    PackMicrokernelTester()
      .mr(4)
      .m(m)
      .k(1)
      .Test(xnn_x32_packx_ukernel_4x__scalar);
  }
}

TEST(X32_PACKX_4X__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    PackMicrokernelTester()
      .mr(4)
      .m(4)
      .k(k)
      .Test(xnn_x32_packx_ukernel_4x__scalar);
  }
}

TEST(X32_PACKX_4X__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (size_t m = 1; m <= 4; m++) {
      PackMicrokernelTester()
        .mr(4)
        .m(m)
        .k(k)
        .Test(xnn_x32_packx_ukernel_4x__scalar);
    }
  }
}

TEST(X32_PACKX_4X__SCALAR, strided_x) {
  for (size_t k = 1; k <= 5; k += 2) {
    PackMicrokernelTester()
      .mr(4)
      .m(4)
      .k(k)
      .x_stride(7)
      .Test(xnn_x32_packx_ukernel_4x__scalar);
  }
}