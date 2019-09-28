// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include <xnnpack/isa-checks.h>
#include <xnnpack/clamp.h>

#include "clamp-microkernel-tester.h"


#if !CPUINFO_ARCH_WASM && !CPUINFO_ARCH_ASMJS
  TEST(F32_CLAMP__PSIMD, n_eq_4) {
    TEST_REQUIRES_PSIMD;
    ClampMicrokernelTester()
      .n(4)
      .Test(xnn_f32_clamp_ukernel__psimd, ClampMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_CLAMP__PSIMD, n_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 256; n += 4) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__psimd, ClampMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_CLAMP__PSIMD, n_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__psimd, ClampMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_CLAMP__PSIMD, n_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__psimd, ClampMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_CLAMP__PSIMD, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 64; n += 3) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_f32_clamp_ukernel__psimd, ClampMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_CLAMP__PSIMD, qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 64; n += 5) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_f32_clamp_ukernel__psimd, ClampMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_CLAMP__PSIMD, qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 64; n += 5) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_f32_clamp_ukernel__psimd, ClampMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // !CPUINFO_ARCH_WASM && !CPUINFO_ARCH_ASMJS


TEST(F32_CLAMP__SCALAR, n_eq_2) {
  ClampMicrokernelTester()
    .n(2)
    .Test(xnn_f32_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
}

TEST(F32_CLAMP__SCALAR, n_div_2) {
  for (size_t n = 4; n < 128; n += 2) {
    ClampMicrokernelTester()
      .n(n)
      .Test(xnn_f32_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_CLAMP__SCALAR, n_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    ClampMicrokernelTester()
      .n(n)
      .Test(xnn_f32_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_CLAMP__SCALAR, n_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    ClampMicrokernelTester()
      .n(n)
      .Test(xnn_f32_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_CLAMP__SCALAR, inplace) {
  for (size_t n = 1; n < 32; n += 3) {
    ClampMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplace(true)
      .Test(xnn_f32_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_CLAMP__SCALAR, qmin) {
  for (size_t n = 1; n < 32; n += 3) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmin(qmin)
        .qmax(255)
        .Test(xnn_f32_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_CLAMP__SCALAR, qmax) {
  for (size_t n = 1; n < 32; n += 3) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmin(0)
        .qmax(qmax)
        .Test(xnn_f32_clamp_ukernel__scalar, ClampMicrokernelTester::Variant::Scalar);
    }
  }
}

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  TEST(F32_CLAMP__NEON, n_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    ClampMicrokernelTester()
      .n(4)
      .Test(xnn_f32_clamp_ukernel__neon);
  }

  TEST(F32_CLAMP__NEON, n_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 256; n += 4) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__neon);
    }
  }

  TEST(F32_CLAMP__NEON, n_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__neon);
    }
  }

  TEST(F32_CLAMP__NEON, n_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__neon);
    }
  }

  TEST(F32_CLAMP__NEON, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 64; n += 3) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_f32_clamp_ukernel__neon);
    }
  }

  TEST(F32_CLAMP__NEON, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 64; n += 5) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_f32_clamp_ukernel__neon);
      }
    }
  }

  TEST(F32_CLAMP__NEON, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 64; n += 5) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_f32_clamp_ukernel__neon);
      }
    }
  }
#endif  // CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(F32_CLAMP__SSE, n_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    ClampMicrokernelTester()
      .n(4)
      .Test(xnn_f32_clamp_ukernel__sse);
  }

  TEST(F32_CLAMP__SSE, n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 256; n += 4) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__sse);
    }
  }

  TEST(F32_CLAMP__SSE, n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__sse);
    }
  }

  TEST(F32_CLAMP__SSE, n_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__sse);
    }
  }

  TEST(F32_CLAMP__SSE, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 64; n += 3) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_f32_clamp_ukernel__sse);
    }
  }

  TEST(F32_CLAMP__SSE, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 64; n += 5) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_f32_clamp_ukernel__sse);
      }
    }
  }

  TEST(F32_CLAMP__SSE, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 64; n += 5) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_f32_clamp_ukernel__sse);
      }
    }
  }
#endif  // CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
