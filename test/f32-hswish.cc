// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/hswish.h>
#include "hswish-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_HSWISH__NEON, n_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    HSwishMicrokernelTester()
      .n(4)
      .Test(xnn_f32_hswish_ukernel__neon);
  }

  TEST(F32_HSWISH__NEON, n_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 256; n += 4) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__neon);
    }
  }

  TEST(F32_HSWISH__NEON, n_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__neon);
    }
  }

  TEST(F32_HSWISH__NEON, n_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__neon);
    }
  }

  TEST(F32_HSWISH__NEON, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 64; n += 3) {
      HSwishMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_f32_hswish_ukernel__neon);
    }
  }

  TEST(F32_HSWISH__NEONFMA, n_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    HSwishMicrokernelTester()
      .n(4)
      .Test(xnn_f32_hswish_ukernel__neonfma);
  }

  TEST(F32_HSWISH__NEONFMA, n_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t n = 4; n < 256; n += 4) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__neonfma);
    }
  }

  TEST(F32_HSWISH__NEONFMA, n_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t n = 5; n < 8; n++) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__neonfma);
    }
  }

  TEST(F32_HSWISH__NEONFMA, n_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t n = 1; n < 4; n++) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__neonfma);
    }
  }

  TEST(F32_HSWISH__NEONFMA, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t n = 1; n < 64; n += 3) {
      HSwishMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_f32_hswish_ukernel__neonfma);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_HSWISH__SSE, n_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    HSwishMicrokernelTester()
      .n(4)
      .Test(xnn_f32_hswish_ukernel__sse);
  }

  TEST(F32_HSWISH__SSE, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 256; n += 4) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__sse);
    }
  }

  TEST(F32_HSWISH__SSE, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__sse);
    }
  }

  TEST(F32_HSWISH__SSE, n_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__sse);
    }
  }

  TEST(F32_HSWISH__SSE, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 64; n += 3) {
      HSwishMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_f32_hswish_ukernel__sse);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_WASM && !XNN_ARCH_ASMJS
  TEST(F32_HSWISH__PSIMD, n_eq_4) {
    TEST_REQUIRES_PSIMD;
    HSwishMicrokernelTester()
      .n(4)
      .Test(xnn_f32_hswish_ukernel__psimd, HSwishMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_HSWISH__PSIMD, n_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 256; n += 4) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__psimd, HSwishMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_HSWISH__PSIMD, n_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__psimd, HSwishMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_HSWISH__PSIMD, n_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      HSwishMicrokernelTester()
        .n(n)
        .Test(xnn_f32_hswish_ukernel__psimd, HSwishMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_HSWISH__PSIMD, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 64; n += 3) {
      HSwishMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_f32_hswish_ukernel__psimd, HSwishMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_WASM && !XNN_ARCH_ASMJS


TEST(F32_HSWISH__SCALAR, n_eq_1) {
  HSwishMicrokernelTester()
    .n(1)
    .Test(xnn_f32_hswish_ukernel__scalar, HSwishMicrokernelTester::Variant::Scalar);
}

TEST(F32_HSWISH__SCALAR, n_gt_1) {
  for (size_t n = 2; n < 8; n++) {
    HSwishMicrokernelTester()
      .n(n)
      .Test(xnn_f32_hswish_ukernel__scalar, HSwishMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_HSWISH__SCALAR, inplace) {
  for (size_t n = 1; n < 32; n += 3) {
    HSwishMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplace(true)
      .Test(xnn_f32_hswish_ukernel__scalar, HSwishMicrokernelTester::Variant::Scalar);
  }
}
