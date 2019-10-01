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

#include <xnnpack/clamp.h>
#include "clamp-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U8_CLAMP__NEON, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    ClampMicrokernelTester()
      .n(8)
      .Test(xnn_u8_clamp_ukernel__neon);
  }

  TEST(U8_CLAMP__NEON, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 512; n += 8) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_u8_clamp_ukernel__neon);
    }
  }

  TEST(U8_CLAMP__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_u8_clamp_ukernel__neon);
    }
  }

  TEST(U8_CLAMP__NEON, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_u8_clamp_ukernel__neon);
    }
  }

  TEST(U8_CLAMP__NEON, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 128; n += 5) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_u8_clamp_ukernel__neon);
    }
  }

  TEST(U8_CLAMP__NEON, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 128; n += 11) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_u8_clamp_ukernel__neon);
      }
    }
  }

  TEST(U8_CLAMP__NEON, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 128; n += 11) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_u8_clamp_ukernel__neon);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(U8_CLAMP__SSE2, n_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    ClampMicrokernelTester()
      .n(8)
      .Test(xnn_u8_clamp_ukernel__sse2);
  }

  TEST(U8_CLAMP__SSE2, n_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 8; n < 512; n += 8) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_u8_clamp_ukernel__sse2);
    }
  }

  TEST(U8_CLAMP__SSE2, n_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_u8_clamp_ukernel__sse2);
    }
  }

  TEST(U8_CLAMP__SSE2, n_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_u8_clamp_ukernel__sse2);
    }
  }

  TEST(U8_CLAMP__SSE2, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 128; n += 5) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_u8_clamp_ukernel__sse2);
    }
  }

  TEST(U8_CLAMP__SSE2, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 128; n += 11) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_u8_clamp_ukernel__sse2);
      }
    }
  }

  TEST(U8_CLAMP__SSE2, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 128; n += 11) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_u8_clamp_ukernel__sse2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

TEST(U8_CLAMP__SCALAR, n_eq_1) {
  ClampMicrokernelTester()
    .n(1)
    .Test(xnn_u8_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
}

TEST(U8_CLAMP__SCALAR, n_gt_1) {
  for (size_t n = 2; n < 8; n++) {
    ClampMicrokernelTester()
      .n(n)
      .Test(xnn_u8_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_CLAMP__SCALAR, inplace) {
  for (size_t n = 1; n < 16; n += 5) {
    ClampMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplace(true)
      .Test(xnn_u8_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_CLAMP__SCALAR, qmin) {
  for (size_t n = 1; n < 16; n += 5) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmin(qmin)
        .qmax(255)
        .Test(xnn_u8_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(U8_CLAMP__SCALAR, qmax) {
  for (size_t n = 1; n < 16; n += 5) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmin(0)
        .qmax(qmax)
        .Test(xnn_u8_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
    }
  }
}
