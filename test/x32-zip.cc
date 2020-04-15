// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/zip.h>
#include "zip-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_ZIP_X2__NEON, n_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    ZipMicrokernelTester()
      .n(4)
      .g(2)
      .Test(xnn_x32_zip_x2_ukernel__neon);
  }

  TEST(X32_ZIP_X2__NEON, n_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x32_zip_x2_ukernel__neon);
    }
  }

  TEST(X32_ZIP_X2__NEON, n_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x32_zip_x2_ukernel__neon);
    }
  }

  TEST(X32_ZIP_X2__NEON, n_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x32_zip_x2_ukernel__neon);
    }
  }

  TEST(X32_ZIP_X3__NEON, n_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    ZipMicrokernelTester()
      .n(4)
      .g(3)
      .Test(xnn_x32_zip_x3_ukernel__neon);
  }

  TEST(X32_ZIP_X3__NEON, n_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x32_zip_x3_ukernel__neon);
    }
  }

  TEST(X32_ZIP_X3__NEON, n_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x32_zip_x3_ukernel__neon);
    }
  }

  TEST(X32_ZIP_X3__NEON, n_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x32_zip_x3_ukernel__neon);
    }
  }

  TEST(X32_ZIP_X4__NEON, n_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    ZipMicrokernelTester()
      .n(4)
      .g(4)
      .Test(xnn_x32_zip_x4_ukernel__neon);
  }

  TEST(X32_ZIP_X4__NEON, n_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_x4_ukernel__neon);
    }
  }

  TEST(X32_ZIP_X4__NEON, n_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_x4_ukernel__neon);
    }
  }

  TEST(X32_ZIP_X4__NEON, n_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_x4_ukernel__neon);
    }
  }

  TEST(X32_ZIP_XM__NEON, n_eq_4_m_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    ZipMicrokernelTester()
      .n(4)
      .g(4)
      .Test(xnn_x32_zip_xm_ukernel__neon);
  }

  TEST(X32_ZIP_XM__NEON, n_eq_4_m_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester()
        .n(4)
        .g(g)
        .Test(xnn_x32_zip_xm_ukernel__neon);
    }
  }

  TEST(X32_ZIP_XM__NEON, n_eq_4_m_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 5; g < 8; g++) {
      ZipMicrokernelTester()
        .n(4)
        .g(g)
        .Test(xnn_x32_zip_xm_ukernel__neon);
    }
  }

  TEST(X32_ZIP_XM__NEON, n_div_4_m_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_xm_ukernel__neon);
    }
  }

  TEST(X32_ZIP_XM__NEON, n_div_4_m_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 4) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__neon);
      }
    }
  }

  TEST(X32_ZIP_XM__NEON, n_div_4_m_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 4) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__neon);
      }
    }
  }

  TEST(X32_ZIP_XM__NEON, n_gt_4_m_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_xm_ukernel__neon);
    }
  }

  TEST(X32_ZIP_XM__NEON, n_gt_4_m_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__neon);
      }
    }
  }

  TEST(X32_ZIP_XM__NEON, n_gt_4_m_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__neon);
      }
    }
  }

  TEST(X32_ZIP_XM__NEON, n_lt_4_m_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_xm_ukernel__neon);
    }
  }

  TEST(X32_ZIP_XM__NEON, n_lt_4_m_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__neon);
      }
    }
  }

  TEST(X32_ZIP_XM__NEON, n_lt_4_m_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__neon);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_ZIP_X2__SSE2, n_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    ZipMicrokernelTester()
      .n(4)
      .g(2)
      .Test(xnn_x32_zip_x2_ukernel__sse2);
  }

  TEST(X32_ZIP_X2__SSE2, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x32_zip_x2_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_X2__SSE2, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x32_zip_x2_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_X2__SSE2, n_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x32_zip_x2_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_X3__SSE2, n_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    ZipMicrokernelTester()
      .n(4)
      .g(3)
      .Test(xnn_x32_zip_x3_ukernel__sse2);
  }

  TEST(X32_ZIP_X3__SSE2, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x32_zip_x3_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_X3__SSE2, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x32_zip_x3_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_X3__SSE2, n_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x32_zip_x3_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_X4__SSE2, n_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    ZipMicrokernelTester()
      .n(4)
      .g(4)
      .Test(xnn_x32_zip_x4_ukernel__sse2);
  }

  TEST(X32_ZIP_X4__SSE2, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_x4_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_X4__SSE2, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_x4_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_X4__SSE2, n_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_x4_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_eq_4_m_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    ZipMicrokernelTester()
      .n(4)
      .g(4)
      .Test(xnn_x32_zip_xm_ukernel__sse2);
  }

  TEST(X32_ZIP_XM__SSE2, n_eq_4_m_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester()
        .n(4)
        .g(g)
        .Test(xnn_x32_zip_xm_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_eq_4_m_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 5; g < 8; g++) {
      ZipMicrokernelTester()
        .n(4)
        .g(g)
        .Test(xnn_x32_zip_xm_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_div_4_m_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_xm_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_div_4_m_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 4) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__sse2);
      }
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_div_4_m_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 4) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__sse2);
      }
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_gt_4_m_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_xm_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_gt_4_m_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__sse2);
      }
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_gt_4_m_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__sse2);
      }
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_lt_4_m_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_xm_ukernel__sse2);
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_lt_4_m_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__sse2);
      }
    }
  }

  TEST(X32_ZIP_XM__SSE2, n_lt_4_m_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__sse2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(X32_ZIP_X2__PSIMD, n_eq_4) {
    TEST_REQUIRES_PSIMD;
    ZipMicrokernelTester()
      .n(4)
      .g(2)
      .Test(xnn_x32_zip_x2_ukernel__psimd);
  }

  TEST(X32_ZIP_X2__PSIMD, n_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x32_zip_x2_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_X2__PSIMD, n_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x32_zip_x2_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_X2__PSIMD, n_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x32_zip_x2_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_X3__PSIMD, n_eq_4) {
    TEST_REQUIRES_PSIMD;
    ZipMicrokernelTester()
      .n(4)
      .g(3)
      .Test(xnn_x32_zip_x3_ukernel__psimd);
  }

  TEST(X32_ZIP_X3__PSIMD, n_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x32_zip_x3_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_X3__PSIMD, n_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x32_zip_x3_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_X3__PSIMD, n_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x32_zip_x3_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_X4__PSIMD, n_eq_4) {
    TEST_REQUIRES_PSIMD;
    ZipMicrokernelTester()
      .n(4)
      .g(4)
      .Test(xnn_x32_zip_x4_ukernel__psimd);
  }

  TEST(X32_ZIP_X4__PSIMD, n_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_x4_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_X4__PSIMD, n_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_x4_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_X4__PSIMD, n_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_x4_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_eq_4_m_eq_4) {
    TEST_REQUIRES_PSIMD;
    ZipMicrokernelTester()
      .n(4)
      .g(4)
      .Test(xnn_x32_zip_xm_ukernel__psimd);
  }

  TEST(X32_ZIP_XM__PSIMD, n_eq_4_m_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester()
        .n(4)
        .g(g)
        .Test(xnn_x32_zip_xm_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_eq_4_m_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t g = 5; g < 8; g++) {
      ZipMicrokernelTester()
        .n(4)
        .g(g)
        .Test(xnn_x32_zip_xm_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_div_4_m_eq_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_xm_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_div_4_m_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 4) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__psimd);
      }
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_div_4_m_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 4) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__psimd);
      }
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_gt_4_m_eq_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_xm_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_gt_4_m_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__psimd);
      }
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_gt_4_m_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__psimd);
      }
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_lt_4_m_eq_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x32_zip_xm_ukernel__psimd);
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_lt_4_m_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__psimd);
      }
    }
  }

  TEST(X32_ZIP_XM__PSIMD, n_lt_4_m_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x32_zip_xm_ukernel__psimd);
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC

TEST(X32_ZIP_X2__SCALAR, n_eq_1) {
  ZipMicrokernelTester()
    .n(1)
    .g(2)
    .Test(xnn_x32_zip_x2_ukernel__scalar);
}

TEST(X32_ZIP_X2__SCALAR, n_gt_1) {
  for (size_t n = 2; n < 8; n++) {
    ZipMicrokernelTester()
      .n(n)
      .g(2)
      .Test(xnn_x32_zip_x2_ukernel__scalar);
  }
}

TEST(X32_ZIP_X3__SCALAR, n_eq_1) {
  ZipMicrokernelTester()
    .n(9)
    .g(3)
    .Test(xnn_x32_zip_x3_ukernel__scalar);
}

TEST(X32_ZIP_X3__SCALAR, n_gt_1) {
  for (size_t n = 2; n < 8; n++) {
    ZipMicrokernelTester()
      .n(n)
      .g(3)
      .Test(xnn_x32_zip_x3_ukernel__scalar);
  }
}

TEST(X32_ZIP_X4__SCALAR, n_eq_1) {
  ZipMicrokernelTester()
    .n(1)
    .g(4)
    .Test(xnn_x32_zip_x4_ukernel__scalar);
}

TEST(X32_ZIP_X4__SCALAR, n_gt_1) {
  for (size_t n = 2; n < 8; n++) {
    ZipMicrokernelTester()
      .n(n)
      .g(4)
      .Test(xnn_x32_zip_x4_ukernel__scalar);
  }
}

TEST(X32_ZIP_XM__SCALAR, n_eq_1_m_eq_4) {
  ZipMicrokernelTester()
    .n(1)
    .g(4)
    .Test(xnn_x32_zip_xm_ukernel__scalar);
}

TEST(X32_ZIP_XM__SCALAR, n_eq_1_m_div_4) {
  for (size_t g = 4; g < 32; g += 4) {
    ZipMicrokernelTester()
      .n(1)
      .g(g)
      .Test(xnn_x32_zip_xm_ukernel__scalar);
  }
}

TEST(X32_ZIP_XM__SCALAR, n_eq_1_m_gt_4) {
  for (size_t g = 5; g < 8; g++) {
    ZipMicrokernelTester()
      .n(1)
      .g(g)
      .Test(xnn_x32_zip_xm_ukernel__scalar);
  }
}

TEST(X32_ZIP_XM__SCALAR, n_gt_1_m_eq_4) {
  for (size_t n = 2; n < 8; n++) {
    ZipMicrokernelTester()
      .n(n)
      .g(4)
      .Test(xnn_x32_zip_xm_ukernel__scalar);
  }
}

TEST(X32_ZIP_XM__SCALAR, n_gt_1_m_div_4) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(g)
        .Test(xnn_x32_zip_xm_ukernel__scalar);
    }
  }
}

TEST(X32_ZIP_XM__SCALAR, n_gt_1_m_gt_4) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t g = 5; g < 8; g++) {
      ZipMicrokernelTester()
        .n(n)
        .g(g)
        .Test(xnn_x32_zip_xm_ukernel__scalar);
    }
  }
}
