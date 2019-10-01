// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/rmax.h>
#include "rmax-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U8RMAX__NEON, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 16; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_u8_rmax_ukernel__neon);
    }
  }

  TEST(U8RMAX__NEON, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RMaxMicrokernelTester()
      .n(16)
      .Test(xnn_u8_rmax_ukernel__neon);
  }

  TEST(U8RMAX__NEON, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 16; n < 128; n += 16) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_u8_rmax_ukernel__neon);
    }
  }

  TEST(U8RMAX__NEON, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 16; n < 32; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_u8_rmax_ukernel__neon);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(U8RMAX__SSE2, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 16; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_u8_rmax_ukernel__sse2);
    }
  }

  TEST(U8RMAX__SSE2, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    RMaxMicrokernelTester()
      .n(16)
      .Test(xnn_u8_rmax_ukernel__sse2);
  }

  TEST(U8RMAX__SSE2, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 16; n < 128; n += 16) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_u8_rmax_ukernel__sse2);
    }
  }

  TEST(U8RMAX__SSE2, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 17; n < 32; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_u8_rmax_ukernel__sse2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

TEST(U8RMAX__SCALAR, n_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    RMaxMicrokernelTester()
      .n(n)
      .Test(xnn_u8_rmax_ukernel__scalar);
  }
}

TEST(U8RMAX__SCALAR, n_eq_2) {
  RMaxMicrokernelTester()
    .n(2)
    .Test(xnn_u8_rmax_ukernel__scalar);
}

TEST(U8RMAX__SCALAR, n_div_2) {
  for (size_t n = 2; n < 16; n += 2) {
    RMaxMicrokernelTester()
      .n(n)
      .Test(xnn_u8_rmax_ukernel__scalar);
  }
}

TEST(U8RMAX__SCALAR, n_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    RMaxMicrokernelTester()
      .n(n)
      .Test(xnn_u8_rmax_ukernel__scalar);
  }
}
