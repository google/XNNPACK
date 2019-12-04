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
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_CLAMP__AVX, n_eq_8) {
    TEST_REQUIRES_X86_AVX;
    ClampMicrokernelTester()
      .n(8)
      .Test(xnn_f32_clamp_ukernel__avx);
  }

  TEST(F32_CLAMP__AVX, n_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 8; n < 512; n += 8) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__avx);
    }
  }

  TEST(F32_CLAMP__AVX, n_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 9; n < 16; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__avx);
    }
  }

  TEST(F32_CLAMP__AVX, n_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 1; n < 8; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__avx);
    }
  }

  TEST(F32_CLAMP__AVX, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 1; n < 128; n += 7) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_f32_clamp_ukernel__avx);
    }
  }

  TEST(F32_CLAMP__AVX, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 1; n < 128; n += 7) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_f32_clamp_ukernel__avx);
      }
    }
  }

  TEST(F32_CLAMP__AVX, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 1; n < 128; n += 7) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_f32_clamp_ukernel__avx);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_CLAMP__AVX512F, n_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    ClampMicrokernelTester()
      .n(16)
      .Test(xnn_f32_clamp_ukernel__avx512f);
  }

  TEST(F32_CLAMP__AVX512F, n_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 16; n < 1024; n += 16) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__avx512f);
    }
  }

  TEST(F32_CLAMP__AVX512F, n_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 17; n < 32; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__avx512f);
    }
  }

  TEST(F32_CLAMP__AVX512F, n_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 1; n < 16; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__avx512f);
    }
  }

  TEST(F32_CLAMP__AVX512F, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 1; n < 256; n += 15) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_f32_clamp_ukernel__avx512f);
    }
  }

  TEST(F32_CLAMP__AVX512F, qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 1; n < 256; n += 15) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_f32_clamp_ukernel__avx512f);
      }
    }
  }

  TEST(F32_CLAMP__AVX512F, qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 1; n < 256; n += 15) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_f32_clamp_ukernel__avx512f);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if !XNN_ARCH_WASM && !XNN_ARCH_ASMJS
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
#endif  // !XNN_ARCH_WASM && !XNN_ARCH_ASMJS


#if XNN_ARCH_WASM
  TEST(F32_CLAMP__WASM, n_eq_2) {
    ClampMicrokernelTester()
      .n(2)
      .Test(xnn_f32_clamp_ukernel__wasm, ClampMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_CLAMP__WASM, n_div_2) {
    for (size_t n = 4; n < 128; n += 2) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__wasm, ClampMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_CLAMP__WASM, n_gt_2) {
    for (size_t n = 3; n < 4; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__wasm, ClampMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_CLAMP__WASM, n_lt_2) {
    for (size_t n = 1; n < 2; n++) {
      ClampMicrokernelTester()
        .n(n)
        .Test(xnn_f32_clamp_ukernel__wasm, ClampMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_CLAMP__WASM, inplace) {
    for (size_t n = 1; n < 32; n += 3) {
      ClampMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace(true)
        .Test(xnn_f32_clamp_ukernel__wasm, ClampMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_CLAMP__WASM, qmin) {
    for (size_t n = 1; n < 32; n += 3) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_f32_clamp_ukernel__wasm, ClampMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_CLAMP__WASM, qmax) {
    for (size_t n = 1; n < 32; n += 3) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .iterations(1)
          .n(n)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_f32_clamp_ukernel__wasm, ClampMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // XNN_ARCH_WASM


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
