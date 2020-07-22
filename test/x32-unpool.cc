// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/unpool.h>
#include "unpool-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_UNPOOL__NEON, c_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    UnpoolMicrokernelTester()
      .p(10)
      .c(4)
      .Test(xnn_x32_unpool_ukernel__neon);
  }

  TEST(X32_UNPOOL__NEON, c_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t c = 8; c < 32; c += 4) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .Test(xnn_x32_unpool_ukernel__neon);
    }
  }

  TEST(X32_UNPOOL__NEON, c_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t c = 1; c < 4; c++) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .Test(xnn_x32_unpool_ukernel__neon);
    }
  }

  TEST(X32_UNPOOL__NEON, c_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t c = 5; c < 8; c++) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(4)
        .Test(xnn_x32_unpool_ukernel__neon);
    }
  }

  TEST(X32_UNPOOL__NEON, varying_p) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t p = 1; p < 20; p += 3) {
      for (size_t c = 1; c < 32; c += 5) {
        UnpoolMicrokernelTester()
          .p(p)
          .c(c)
          .Test(xnn_x32_unpool_ukernel__neon);
      }
    }
  }

  TEST(X32_UNPOOL__NEON, varying_f) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t c = 1; c < 32; c += 5) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .f(0xDEADBEAF)
        .Test(xnn_x32_unpool_ukernel__neon);
    }
  }

  TEST(X32_UNPOOL__NEON, y_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t c = 1; c < 32; c += 5) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .y_stride(c * 2 + 7)
        .Test(xnn_x32_unpool_ukernel__neon);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_UNPOOL__SSE2, c_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    UnpoolMicrokernelTester()
      .p(10)
      .c(4)
      .Test(xnn_x32_unpool_ukernel__sse2);
  }

  TEST(X32_UNPOOL__SSE2, c_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t c = 8; c < 32; c += 4) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .Test(xnn_x32_unpool_ukernel__sse2);
    }
  }

  TEST(X32_UNPOOL__SSE2, c_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t c = 1; c < 4; c++) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .Test(xnn_x32_unpool_ukernel__sse2);
    }
  }

  TEST(X32_UNPOOL__SSE2, c_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t c = 5; c < 8; c++) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(4)
        .Test(xnn_x32_unpool_ukernel__sse2);
    }
  }

  TEST(X32_UNPOOL__SSE2, varying_p) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t p = 1; p < 20; p += 3) {
      for (size_t c = 1; c < 32; c += 5) {
        UnpoolMicrokernelTester()
          .p(p)
          .c(c)
          .Test(xnn_x32_unpool_ukernel__sse2);
      }
    }
  }

  TEST(X32_UNPOOL__SSE2, varying_f) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t c = 1; c < 32; c += 5) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .f(0xDEADBEAF)
        .Test(xnn_x32_unpool_ukernel__sse2);
    }
  }

  TEST(X32_UNPOOL__SSE2, y_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t c = 1; c < 32; c += 5) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .y_stride(c * 2 + 7)
        .Test(xnn_x32_unpool_ukernel__sse2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  TEST(X32_UNPOOL__WASMSIMD, c_eq_4) {
    UnpoolMicrokernelTester()
      .p(10)
      .c(4)
      .Test(xnn_x32_unpool_ukernel__wasmsimd);
  }

  TEST(X32_UNPOOL__WASMSIMD, c_div_4) {
    for (size_t c = 8; c < 32; c += 4) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .Test(xnn_x32_unpool_ukernel__wasmsimd);
    }
  }

  TEST(X32_UNPOOL__WASMSIMD, c_lt_4) {
    for (size_t c = 1; c < 4; c++) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .Test(xnn_x32_unpool_ukernel__wasmsimd);
    }
  }

  TEST(X32_UNPOOL__WASMSIMD, c_gt_4) {
    for (size_t c = 5; c < 8; c++) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(4)
        .Test(xnn_x32_unpool_ukernel__wasmsimd);
    }
  }

  TEST(X32_UNPOOL__WASMSIMD, varying_p) {
    for (size_t p = 1; p < 20; p += 3) {
      for (size_t c = 1; c < 32; c += 5) {
        UnpoolMicrokernelTester()
          .p(p)
          .c(c)
          .Test(xnn_x32_unpool_ukernel__wasmsimd);
      }
    }
  }

  TEST(X32_UNPOOL__WASMSIMD, varying_f) {
    for (size_t c = 1; c < 32; c += 5) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .f(0xDEADBEAF)
        .Test(xnn_x32_unpool_ukernel__wasmsimd);
    }
  }

  TEST(X32_UNPOOL__WASMSIMD, y_stride) {
    for (size_t c = 1; c < 32; c += 5) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .y_stride(c * 2 + 7)
        .Test(xnn_x32_unpool_ukernel__wasmsimd);
    }
  }
#endif  // XNN_ARCH_WASMSIMD


TEST(X32_UNPOOL__SCALAR, c_eq_1) {
  UnpoolMicrokernelTester()
    .p(10)
    .c(1)
    .Test(xnn_x32_unpool_ukernel__scalar);
}

TEST(X32_UNPOOL__SCALAR, c_gt_1) {
  for (size_t c = 2; c < 16; c++) {
    UnpoolMicrokernelTester()
      .p(10)
      .c(c)
      .Test(xnn_x32_unpool_ukernel__scalar);
  }
}

TEST(X32_UNPOOL__SCALAR, varying_p) {
  for (size_t p = 1; p < 20; p += 3) {
    for (size_t c = 1; c < 16; c += 3) {
      UnpoolMicrokernelTester()
        .p(p)
        .c(c)
        .Test(xnn_x32_unpool_ukernel__scalar);
    }
  }
}

TEST(X32_UNPOOL__SCALAR, varying_f) {
  for (size_t c = 1; c < 16; c += 3) {
    UnpoolMicrokernelTester()
      .p(10)
      .c(c)
      .f(0xDEADBEAF)
      .Test(xnn_x32_unpool_ukernel__scalar);
  }
}

TEST(X32_UNPOOL__SCALAR, y_stride) {
  for (size_t c = 1; c < 16; c += 3) {
    UnpoolMicrokernelTester()
      .p(10)
      .c(c)
      .y_stride(c * 2 + 7)
      .Test(xnn_x32_unpool_ukernel__scalar);
  }
}
