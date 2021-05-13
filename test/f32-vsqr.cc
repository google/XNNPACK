// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vsqr.yaml
//   Generator: tools/generate-vunary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunary.h>
#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQR__NEON_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqr_ukernel__neon_x4, VUnaryMicrokernelTester::OpType::Square);
  }

  TEST(F32_VSQR__NEON_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__neon_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__NEON_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__neon_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__NEON_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__neon_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__NEON_X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqr_ukernel__neon_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQR__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqr_ukernel__neon_x8, VUnaryMicrokernelTester::OpType::Square);
  }

  TEST(F32_VSQR__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__neon_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__neon_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__neon_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__NEON_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqr_ukernel__neon_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQR__SSE_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqr_ukernel__sse_x4, VUnaryMicrokernelTester::OpType::Square);
  }

  TEST(F32_VSQR__SSE_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__sse_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__SSE_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__sse_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__SSE_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__sse_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__SSE_X4, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqr_ukernel__sse_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQR__SSE_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqr_ukernel__sse_x8, VUnaryMicrokernelTester::OpType::Square);
  }

  TEST(F32_VSQR__SSE_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__sse_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__SSE_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__sse_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__SSE_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__sse_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__SSE_X8, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqr_ukernel__sse_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQR__AVX_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqr_ukernel__avx_x8, VUnaryMicrokernelTester::OpType::Square);
  }

  TEST(F32_VSQR__AVX_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqr_ukernel__avx_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQR__AVX_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqr_ukernel__avx_x16, VUnaryMicrokernelTester::OpType::Square);
  }

  TEST(F32_VSQR__AVX_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx_x16, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx_x16, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx_x16, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqr_ukernel__avx_x16, VUnaryMicrokernelTester::OpType::Square);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQR__AVX512F_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqr_ukernel__avx512f_x16, VUnaryMicrokernelTester::OpType::Square);
  }

  TEST(F32_VSQR__AVX512F_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx512f_x16, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX512F_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx512f_x16, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX512F_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx512f_x16, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX512F_X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqr_ukernel__avx512f_x16, VUnaryMicrokernelTester::OpType::Square);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQR__AVX512F_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsqr_ukernel__avx512f_x32, VUnaryMicrokernelTester::OpType::Square);
  }

  TEST(F32_VSQR__AVX512F_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx512f_x32, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX512F_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx512f_x32, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX512F_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__avx512f_x32, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__AVX512F_X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqr_ukernel__avx512f_x32, VUnaryMicrokernelTester::OpType::Square);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  TEST(F32_VSQR__WASMSIMD_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqr_ukernel__wasmsimd_x4, VUnaryMicrokernelTester::OpType::Square);
  }

  TEST(F32_VSQR__WASMSIMD_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__wasmsimd_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__WASMSIMD_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__wasmsimd_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__WASMSIMD_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__wasmsimd_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__WASMSIMD_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqr_ukernel__wasmsimd_x4, VUnaryMicrokernelTester::OpType::Square);
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_VSQR__WASMSIMD_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqr_ukernel__wasmsimd_x8, VUnaryMicrokernelTester::OpType::Square);
  }

  TEST(F32_VSQR__WASMSIMD_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__wasmsimd_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__WASMSIMD_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__wasmsimd_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__WASMSIMD_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqr_ukernel__wasmsimd_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }

  TEST(F32_VSQR__WASMSIMD_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqr_ukernel__wasmsimd_x8, VUnaryMicrokernelTester::OpType::Square);
    }
  }
#endif  // XNN_ARCH_WASMSIMD


TEST(F32_VSQR__SCALAR_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsqr_ukernel__scalar_x1, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
}

TEST(F32_VSQR__SCALAR_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqr_ukernel__scalar_x1, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSQR__SCALAR_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsqr_ukernel__scalar_x1, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
  }
}


TEST(F32_VSQR__SCALAR_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsqr_ukernel__scalar_x2, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
}

TEST(F32_VSQR__SCALAR_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqr_ukernel__scalar_x2, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSQR__SCALAR_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqr_ukernel__scalar_x2, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSQR__SCALAR_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqr_ukernel__scalar_x2, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSQR__SCALAR_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsqr_ukernel__scalar_x2, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
  }
}


TEST(F32_VSQR__SCALAR_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsqr_ukernel__scalar_x4, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
}

TEST(F32_VSQR__SCALAR_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqr_ukernel__scalar_x4, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSQR__SCALAR_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqr_ukernel__scalar_x4, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSQR__SCALAR_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqr_ukernel__scalar_x4, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSQR__SCALAR_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsqr_ukernel__scalar_x4, VUnaryMicrokernelTester::OpType::Square, VUnaryMicrokernelTester::Variant::Scalar);
  }
}
