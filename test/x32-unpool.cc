// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/unpool.h>
#include "unpool-microkernel-tester.h"


#if !XNN_ARCH_WASM && !XNN_ARCH_ASMJS
  TEST(X32_UNPOOL__PSIMD, c_eq_4) {
    TEST_REQUIRES_PSIMD;
    UnpoolMicrokernelTester()
      .p(10)
      .c(4)
      .Test(xnn_x32_unpool_ukernel__psimd);
  }

  TEST(X32_UNPOOL__PSIMD, c_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t c = 8; c < 32; c += 4) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .Test(xnn_x32_unpool_ukernel__psimd);
    }
  }

  TEST(X32_UNPOOL__PSIMD, c_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t c = 1; c < 4; c++) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .Test(xnn_x32_unpool_ukernel__psimd);
    }
  }

  TEST(X32_UNPOOL__PSIMD, c_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t c = 5; c < 8; c++) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(4)
        .Test(xnn_x32_unpool_ukernel__psimd);
    }
  }

  TEST(X32_UNPOOL__PSIMD, varying_p) {
    TEST_REQUIRES_PSIMD;
    for (size_t p = 1; p < 20; p += 3) {
      for (size_t c = 1; c < 32; c += 5) {
        UnpoolMicrokernelTester()
          .p(p)
          .c(c)
          .Test(xnn_x32_unpool_ukernel__psimd);
      }
    }
  }

  TEST(X32_UNPOOL__PSIMD, varying_f) {
    TEST_REQUIRES_PSIMD;
    for (size_t c = 1; c < 32; c += 5) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .f(0xDEADBEAF)
        .Test(xnn_x32_unpool_ukernel__psimd);
    }
  }

  TEST(X32_UNPOOL__PSIMD, y_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t c = 1; c < 32; c += 5) {
      UnpoolMicrokernelTester()
        .p(10)
        .c(c)
        .y_stride(c * 2 + 7)
        .Test(xnn_x32_unpool_ukernel__psimd);
    }
  }
#endif  // !XNN_ARCH_WASM && !XNN_ARCH_ASMJS


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
