// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/zip.h"
#include "zip-microkernel-tester.h"

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X8_ZIP_X2__NEON, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    ZipMicrokernelTester()
      .n(8)
      .g(2)
      .Test(xnn_x8_zip_x2_ukernel__neon);
  }

  TEST(X8_ZIP_X2__NEON, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 8) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x8_zip_x2_ukernel__neon);
    }
  }

  TEST(X8_ZIP_X2__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x8_zip_x2_ukernel__neon);
    }
  }

  TEST(X8_ZIP_X2__NEON, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x8_zip_x2_ukernel__neon);
    }
  }

  TEST(X8_ZIP_X3__NEON, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    ZipMicrokernelTester()
      .n(9)
      .g(3)
      .Test(xnn_x8_zip_x3_ukernel__neon);
  }

  TEST(X8_ZIP_X3__NEON, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 8) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x8_zip_x3_ukernel__neon);
    }
  }

  TEST(X8_ZIP_X3__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x8_zip_x3_ukernel__neon);
    }
  }

  TEST(X8_ZIP_X3__NEON, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x8_zip_x3_ukernel__neon);
    }
  }

  TEST(X8_ZIP_X4__NEON, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    ZipMicrokernelTester()
      .n(8)
      .g(4)
      .Test(xnn_x8_zip_x4_ukernel__neon);
  }

  TEST(X8_ZIP_X4__NEON, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 8) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x8_zip_x4_ukernel__neon);
    }
  }

  TEST(X8_ZIP_X4__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x8_zip_x4_ukernel__neon);
    }
  }

  TEST(X8_ZIP_X4__NEON, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 16; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x8_zip_x4_ukernel__neon);
    }
  }

  TEST(X8_ZIP_XM__NEON, n_eq_8_m_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    ZipMicrokernelTester()
      .n(8)
      .g(4)
      .Test(xnn_x8_zip_xm_ukernel__neon);
  }

  TEST(X8_ZIP_XM__NEON, n_eq_8_m_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester()
        .n(8)
        .g(g)
        .Test(xnn_x8_zip_xm_ukernel__neon);
    }
  }

  TEST(X8_ZIP_XM__NEON, n_eq_8_m_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 5; g < 8; g++) {
      ZipMicrokernelTester()
        .n(8)
        .g(g)
        .Test(xnn_x8_zip_xm_ukernel__neon);
    }
  }

  TEST(X8_ZIP_XM__NEON, n_div_8_m_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 8) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x8_zip_xm_ukernel__neon);
    }
  }

  TEST(X8_ZIP_XM__NEON, n_div_8_m_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 8) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x8_zip_xm_ukernel__neon);
      }
    }
  }

  TEST(X8_ZIP_XM__NEON, n_div_8_m_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 8) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x8_zip_xm_ukernel__neon);
      }
    }
  }

  TEST(X8_ZIP_XM__NEON, n_gt_8_m_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x8_zip_xm_ukernel__neon);
    }
  }

  TEST(X8_ZIP_XM__NEON, n_gt_8_m_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x8_zip_xm_ukernel__neon);
      }
    }
  }

  TEST(X8_ZIP_XM__NEON, n_gt_8_m_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x8_zip_xm_ukernel__neon);
      }
    }
  }

  TEST(X8_ZIP_XM__NEON, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      for (size_t g = 4; g < 12; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x8_zip_xm_ukernel__neon);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_ZIP_X2__SSE2, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    ZipMicrokernelTester()
      .n(16)
      .g(2)
      .Test(xnn_x8_zip_x2_ukernel__sse2);
  }

  TEST(X8_ZIP_X2__SSE2, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 16; n < 256; n += 16) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x8_zip_x2_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_X2__SSE2, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 17; n < 32; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x8_zip_x2_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_X2__SSE2, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 16; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(2)
        .Test(xnn_x8_zip_x2_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_X3__SSE2, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    ZipMicrokernelTester()
      .n(16)
      .g(3)
      .Test(xnn_x8_zip_x3_ukernel__sse2);
  }

  TEST(X8_ZIP_X3__SSE2, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 16; n < 256; n += 16) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x8_zip_x3_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_X3__SSE2, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 17; n < 32; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x8_zip_x3_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_X3__SSE2, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 16; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(3)
        .Test(xnn_x8_zip_x3_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_X4__SSE2, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    ZipMicrokernelTester()
      .n(16)
      .g(4)
      .Test(xnn_x8_zip_x4_ukernel__sse2);
  }

  TEST(X8_ZIP_X4__SSE2, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 16; n < 256; n += 16) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x8_zip_x4_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_X4__SSE2, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 17; n < 32; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x8_zip_x4_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_X4__SSE2, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 16; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x8_zip_x4_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_eq_8_m_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    ZipMicrokernelTester()
      .n(8)
      .g(4)
      .Test(xnn_x8_zip_xm_ukernel__sse2);
  }

  TEST(X8_ZIP_XM__SSE2, n_eq_8_m_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester()
        .n(8)
        .g(g)
        .Test(xnn_x8_zip_xm_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_eq_8_m_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 5; g < 8; g++) {
      ZipMicrokernelTester()
        .n(8)
        .g(g)
        .Test(xnn_x8_zip_xm_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_eq_16_m_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    ZipMicrokernelTester()
      .n(16)
      .g(4)
      .Test(xnn_x8_zip_xm_ukernel__sse2);
  }

  TEST(X8_ZIP_XM__SSE2, n_eq_16_m_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester()
        .n(16)
        .g(g)
        .Test(xnn_x8_zip_xm_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_eq_16_m_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 5; g < 8; g++) {
      ZipMicrokernelTester()
        .n(16)
        .g(g)
        .Test(xnn_x8_zip_xm_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_div_16_m_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 16; n < 256; n += 16) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x8_zip_xm_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_div_16_m_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 16; n < 256; n += 16) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x8_zip_xm_ukernel__sse2);
      }
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_div_16_m_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 16; n < 256; n += 16) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x8_zip_xm_ukernel__sse2);
      }
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_gt_16_m_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 17; n < 32; n++) {
      ZipMicrokernelTester()
        .n(n)
        .g(4)
        .Test(xnn_x8_zip_xm_ukernel__sse2);
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_gt_16_m_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 17; n < 32; n++) {
      for (size_t g = 4; g < 32; g += 4) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x8_zip_xm_ukernel__sse2);
      }
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_gt_16_m_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 17; n < 32; n++) {
      for (size_t g = 5; g < 8; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x8_zip_xm_ukernel__sse2);
      }
    }
  }

  TEST(X8_ZIP_XM__SSE2, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 16; n++) {
      for (size_t g = 4; g < 12; g++) {
        ZipMicrokernelTester()
          .n(n)
          .g(g)
          .Test(xnn_x8_zip_xm_ukernel__sse2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

TEST(X8_ZIP_X2__SCALAR, n_eq_1) {
  ZipMicrokernelTester()
    .n(1)
    .g(2)
    .Test(xnn_x8_zip_x2_ukernel__scalar);
}

TEST(X8_ZIP_X2__SCALAR, n_gt_1) {
  for (size_t n = 2; n < 8; n++) {
    ZipMicrokernelTester()
      .n(n)
      .g(2)
      .Test(xnn_x8_zip_x2_ukernel__scalar);
  }
}

TEST(X8_ZIP_X3__SCALAR, n_eq_1) {
  ZipMicrokernelTester()
    .n(9)
    .g(3)
    .Test(xnn_x8_zip_x3_ukernel__scalar);
}

TEST(X8_ZIP_X3__SCALAR, n_gt_1) {
  for (size_t n = 2; n < 8; n++) {
    ZipMicrokernelTester()
      .n(n)
      .g(3)
      .Test(xnn_x8_zip_x3_ukernel__scalar);
  }
}

TEST(X8_ZIP_X4__SCALAR, n_eq_1) {
  ZipMicrokernelTester()
    .n(1)
    .g(4)
    .Test(xnn_x8_zip_x4_ukernel__scalar);
}

TEST(X8_ZIP_X4__SCALAR, n_gt_1) {
  for (size_t n = 2; n < 8; n++) {
    ZipMicrokernelTester()
      .n(n)
      .g(4)
      .Test(xnn_x8_zip_x4_ukernel__scalar);
  }
}

TEST(X8_ZIP_XM__SCALAR, n_eq_1_m_eq_4) {
  ZipMicrokernelTester()
    .n(1)
    .g(4)
    .Test(xnn_x8_zip_xm_ukernel__scalar);
}

TEST(X8_ZIP_XM__SCALAR, n_eq_1_m_div_4) {
  for (size_t g = 4; g < 32; g += 4) {
    ZipMicrokernelTester()
      .n(1)
      .g(g)
      .Test(xnn_x8_zip_xm_ukernel__scalar);
  }
}

TEST(X8_ZIP_XM__SCALAR, n_eq_1_m_gt_4) {
  for (size_t g = 5; g < 8; g++) {
    ZipMicrokernelTester()
      .n(1)
      .g(g)
      .Test(xnn_x8_zip_xm_ukernel__scalar);
  }
}

TEST(X8_ZIP_XM__SCALAR, n_gt_1_m_eq_4) {
  for (size_t n = 2; n < 8; n++) {
    ZipMicrokernelTester()
      .n(n)
      .g(4)
      .Test(xnn_x8_zip_xm_ukernel__scalar);
  }
}

TEST(X8_ZIP_XM__SCALAR, n_gt_1_m_div_4) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester()
        .n(n)
        .g(g)
        .Test(xnn_x8_zip_xm_ukernel__scalar);
    }
  }
}

TEST(X8_ZIP_XM__SCALAR, n_gt_1_m_gt_4) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t g = 5; g < 8; g++) {
      ZipMicrokernelTester()
        .n(n)
        .g(g)
        .Test(xnn_x8_zip_xm_ukernel__scalar);
    }
  }
}
