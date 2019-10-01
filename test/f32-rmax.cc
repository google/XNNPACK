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
  TEST(F32_RMAX__NEON, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 16; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__neon);
    }
  }

  TEST(F32_RMAX__NEON, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RMaxMicrokernelTester()
      .n(16)
      .Test(xnn_f32_rmax_ukernel__neon);
  }

  TEST(F32_RMAX__NEON, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 16; n < 128; n += 16) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__neon);
    }
  }

  TEST(F32_RMAX__NEON, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 16; n < 32; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__neon);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMAX__SSE, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 16; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__sse);
    }
  }

  TEST(F32_RMAX__SSE, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    RMaxMicrokernelTester()
      .n(16)
      .Test(xnn_f32_rmax_ukernel__sse);
  }

  TEST(F32_RMAX__SSE, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 16; n < 128; n += 16) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__sse);
    }
  }

  TEST(F32_RMAX__SSE, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 16; n < 32; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__sse);
    }
  }

  TEST(F32_RMAX__AVX, n_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 1; n < 32; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__avx);
    }
  }

  TEST(F32_RMAX__AVX, n_eq_32) {
    TEST_REQUIRES_X86_AVX;
    RMaxMicrokernelTester()
      .n(32)
      .Test(xnn_f32_rmax_ukernel__avx);
  }

  TEST(F32_RMAX__AVX, n_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 32; n < 256; n += 32) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__avx);
    }
  }

  TEST(F32_RMAX__AVX, n_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 32; n < 64; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__avx);
    }
  }

  TEST(F32_RMAX__AVX512F, n_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 1; n < 64; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__avx512f);
    }
  }

  TEST(F32_RMAX__AVX512F, n_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    RMaxMicrokernelTester()
      .n(64)
      .Test(xnn_f32_rmax_ukernel__avx512f);
  }

  TEST(F32_RMAX__AVX512F, n_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 64; n < 512; n += 64) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__avx512f);
    }
  }

  TEST(F32_RMAX__AVX512F, n_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 64; n < 128; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f32_rmax_ukernel__avx512f);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

TEST(F32_RMAX__SCALAR, n_lt_4) {
  for (size_t n = 1; n < 4; n++) {
    RMaxMicrokernelTester()
      .n(n)
      .Test(xnn_f32_rmax_ukernel__scalar);
  }
}

TEST(F32_RMAX__SCALAR, n_eq_4) {
  RMaxMicrokernelTester()
    .n(4)
    .Test(xnn_f32_rmax_ukernel__scalar);
}

TEST(F32_RMAX__SCALAR, n_div_4) {
  for (size_t n = 4; n < 32; n += 4) {
    RMaxMicrokernelTester()
      .n(n)
      .Test(xnn_f32_rmax_ukernel__scalar);
  }
}

TEST(F32_RMAX__SCALAR, n_gt_4) {
  for (size_t n = 4; n < 8; n++) {
    RMaxMicrokernelTester()
      .n(n)
      .Test(xnn_f32_rmax_ukernel__scalar);
  }
}
