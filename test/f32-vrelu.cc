// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vrelu.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VRELU__NEON_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vrelu_ukernel__neon_u4);
  }

  TEST(F32_VRELU__NEON_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__neon_u4);
    }
  }

  TEST(F32_VRELU__NEON_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__neon_u4);
    }
  }

  TEST(F32_VRELU__NEON_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__neon_u4);
    }
  }

  TEST(F32_VRELU__NEON_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__neon_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VRELU__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrelu_ukernel__neon_u8);
  }

  TEST(F32_VRELU__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__neon_u8);
    }
  }

  TEST(F32_VRELU__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__neon_u8);
    }
  }

  TEST(F32_VRELU__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__neon_u8);
    }
  }

  TEST(F32_VRELU__NEON_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__neon_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRELU__SSE_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vrelu_ukernel__sse_u4);
  }

  TEST(F32_VRELU__SSE_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__sse_u4);
    }
  }

  TEST(F32_VRELU__SSE_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__sse_u4);
    }
  }

  TEST(F32_VRELU__SSE_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__sse_u4);
    }
  }

  TEST(F32_VRELU__SSE_U4, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__sse_u4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRELU__SSE_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrelu_ukernel__sse_u8);
  }

  TEST(F32_VRELU__SSE_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__sse_u8);
    }
  }

  TEST(F32_VRELU__SSE_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__sse_u8);
    }
  }

  TEST(F32_VRELU__SSE_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__sse_u8);
    }
  }

  TEST(F32_VRELU__SSE_U8, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__sse_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRELU__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrelu_ukernel__avx_u8);
  }

  TEST(F32_VRELU__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx_u8);
    }
  }

  TEST(F32_VRELU__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx_u8);
    }
  }

  TEST(F32_VRELU__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx_u8);
    }
  }

  TEST(F32_VRELU__AVX_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__avx_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRELU__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vrelu_ukernel__avx_u16);
  }

  TEST(F32_VRELU__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx_u16);
    }
  }

  TEST(F32_VRELU__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx_u16);
    }
  }

  TEST(F32_VRELU__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx_u16);
    }
  }

  TEST(F32_VRELU__AVX_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__avx_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRELU__AVX512F_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vrelu_ukernel__avx512f_u16);
  }

  TEST(F32_VRELU__AVX512F_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx512f_u16);
    }
  }

  TEST(F32_VRELU__AVX512F_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx512f_u16);
    }
  }

  TEST(F32_VRELU__AVX512F_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx512f_u16);
    }
  }

  TEST(F32_VRELU__AVX512F_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__avx512f_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRELU__AVX512F_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vrelu_ukernel__avx512f_u32);
  }

  TEST(F32_VRELU__AVX512F_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx512f_u32);
    }
  }

  TEST(F32_VRELU__AVX512F_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx512f_u32);
    }
  }

  TEST(F32_VRELU__AVX512F_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__avx512f_u32);
    }
  }

  TEST(F32_VRELU__AVX512F_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__avx512f_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(F32_VRELU__SCALAR_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vrelu_ukernel__scalar_u1);
}

TEST(F32_VRELU__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrelu_ukernel__scalar_u1);
  }
}

TEST(F32_VRELU__SCALAR_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vrelu_ukernel__scalar_u1);
  }
}


TEST(F32_VRELU__SCALAR_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vrelu_ukernel__scalar_u2);
}

TEST(F32_VRELU__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrelu_ukernel__scalar_u2);
  }
}

TEST(F32_VRELU__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrelu_ukernel__scalar_u2);
  }
}

TEST(F32_VRELU__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrelu_ukernel__scalar_u2);
  }
}

TEST(F32_VRELU__SCALAR_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vrelu_ukernel__scalar_u2);
  }
}


TEST(F32_VRELU__SCALAR_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vrelu_ukernel__scalar_u4);
}

TEST(F32_VRELU__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrelu_ukernel__scalar_u4);
  }
}

TEST(F32_VRELU__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrelu_ukernel__scalar_u4);
  }
}

TEST(F32_VRELU__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrelu_ukernel__scalar_u4);
  }
}

TEST(F32_VRELU__SCALAR_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vrelu_ukernel__scalar_u4);
  }
}


TEST(F32_VRELU__SCALAR_U8, batch_eq_8) {
  VUnaryMicrokernelTester()
    .batch_size(8)
    .Test(xnn_f32_vrelu_ukernel__scalar_u8);
}

TEST(F32_VRELU__SCALAR_U8, batch_div_8) {
  for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrelu_ukernel__scalar_u8);
  }
}

TEST(F32_VRELU__SCALAR_U8, batch_lt_8) {
  for (size_t batch_size = 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrelu_ukernel__scalar_u8);
  }
}

TEST(F32_VRELU__SCALAR_U8, batch_gt_8) {
  for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrelu_ukernel__scalar_u8);
  }
}

TEST(F32_VRELU__SCALAR_U8, inplace) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vrelu_ukernel__scalar_u8);
  }
}


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRELU__WASM_U1, batch_eq_1) {
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_vrelu_ukernel__wasm_u1);
  }

  TEST(F32_VRELU__WASM_U1, batch_gt_1) {
    for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm_u1);
    }
  }

  TEST(F32_VRELU__WASM_U1, inplace) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__wasm_u1);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRELU__WASM_U2, batch_eq_2) {
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_vrelu_ukernel__wasm_u2);
  }

  TEST(F32_VRELU__WASM_U2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm_u2);
    }
  }

  TEST(F32_VRELU__WASM_U2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm_u2);
    }
  }

  TEST(F32_VRELU__WASM_U2, batch_gt_2) {
    for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm_u2);
    }
  }

  TEST(F32_VRELU__WASM_U2, inplace) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__wasm_u2);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRELU__WASM_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vrelu_ukernel__wasm_u4);
  }

  TEST(F32_VRELU__WASM_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm_u4);
    }
  }

  TEST(F32_VRELU__WASM_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm_u4);
    }
  }

  TEST(F32_VRELU__WASM_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm_u4);
    }
  }

  TEST(F32_VRELU__WASM_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__wasm_u4);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRELU__WASM_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrelu_ukernel__wasm_u8);
  }

  TEST(F32_VRELU__WASM_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm_u8);
    }
  }

  TEST(F32_VRELU__WASM_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm_u8);
    }
  }

  TEST(F32_VRELU__WASM_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm_u8);
    }
  }

  TEST(F32_VRELU__WASM_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__wasm_u8);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRELU__WASMSIMD_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vrelu_ukernel__wasmsimd_u4);
  }

  TEST(F32_VRELU__WASMSIMD_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VRELU__WASMSIMD_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VRELU__WASMSIMD_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VRELU__WASMSIMD_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u4);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRELU__WASMSIMD_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrelu_ukernel__wasmsimd_u8);
  }

  TEST(F32_VRELU__WASMSIMD_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VRELU__WASMSIMD_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VRELU__WASMSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VRELU__WASMSIMD_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u8);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRELU__WASMSIMD_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vrelu_ukernel__wasmsimd_u16);
  }

  TEST(F32_VRELU__WASMSIMD_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u16);
    }
  }

  TEST(F32_VRELU__WASMSIMD_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u16);
    }
  }

  TEST(F32_VRELU__WASMSIMD_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u16);
    }
  }

  TEST(F32_VRELU__WASMSIMD_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__wasmsimd_u16);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_VRELU__WASM32_SHR_U1, batch_eq_1) {
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u1);
  }

  TEST(F32_VRELU__WASM32_SHR_U1, batch_gt_1) {
    for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u1);
    }
  }

  TEST(F32_VRELU__WASM32_SHR_U1, inplace) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u1);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_VRELU__WASM32_SHR_U2, batch_eq_2) {
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u2);
  }

  TEST(F32_VRELU__WASM32_SHR_U2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u2);
    }
  }

  TEST(F32_VRELU__WASM32_SHR_U2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u2);
    }
  }

  TEST(F32_VRELU__WASM32_SHR_U2, batch_gt_2) {
    for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u2);
    }
  }

  TEST(F32_VRELU__WASM32_SHR_U2, inplace) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u2);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_VRELU__WASM32_SHR_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u4);
  }

  TEST(F32_VRELU__WASM32_SHR_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u4);
    }
  }

  TEST(F32_VRELU__WASM32_SHR_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u4);
    }
  }

  TEST(F32_VRELU__WASM32_SHR_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u4);
    }
  }

  TEST(F32_VRELU__WASM32_SHR_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrelu_ukernel__wasm32_shr_u4);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
