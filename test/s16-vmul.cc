// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s16-vmul.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"
#include "vbinary-microkernel-tester.h"


TEST(S16_VMUL__SCALAR_U1, batch_eq_1) {
  VBinaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_s16_vmul_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
}

TEST(S16_VMUL__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s16_vmul_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U1, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U1, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U1, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}


TEST(S16_VMUL__SCALAR_U2, batch_eq_2) {
  VBinaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_s16_vmul_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
}

TEST(S16_VMUL__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s16_vmul_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s16_vmul_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s16_vmul_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U2, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U2, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U2, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}


TEST(S16_VMUL__SCALAR_U4, batch_eq_4) {
  VBinaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_s16_vmul_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
}

TEST(S16_VMUL__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s16_vmul_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s16_vmul_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s16_vmul_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U4, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U4, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U4, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}


TEST(S16_VMUL__SCALAR_U8, batch_eq_8) {
  VBinaryMicrokernelTester()
    .batch_size(8)
    .Test(xnn_s16_vmul_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
}

TEST(S16_VMUL__SCALAR_U8, batch_div_8) {
  for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s16_vmul_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U8, batch_lt_8) {
  for (size_t batch_size = 1; batch_size < 8; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s16_vmul_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U8, batch_gt_8) {
  for (size_t batch_size = 9; batch_size < 16; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s16_vmul_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U8, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U8, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}

TEST(S16_VMUL__SCALAR_U8, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_s16_vmul_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }
}


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S16_VMUL__SSE41_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_s16_vmul_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMUL__SSE41_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__SSE41_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__SSE41_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__SSE41_U8, inplace_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_s16_vmul_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__SSE41_U8, inplace_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__SSE41_U8, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S16_VMUL__SSE41_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_s16_vmul_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMUL__SSE41_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__SSE41_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__SSE41_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__SSE41_U16, inplace_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_s16_vmul_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__SSE41_U16, inplace_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__SSE41_U16, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S16_VMUL__AVX2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_s16_vmul_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMUL__AVX2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U16, inplace_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U16, inplace_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U16, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S16_VMUL__AVX2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VBinaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_s16_vmul_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMUL__AVX2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U32, inplace_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U32, inplace_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U32, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S16_VMUL__WASMSIMD_U8, batch_eq_8) {
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_s16_vmul_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMUL__WASMSIMD_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__WASMSIMD_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__WASMSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__WASMSIMD_U8, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__WASMSIMD_U8, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__WASMSIMD_U8, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S16_VMUL__WASMSIMD_U16, batch_eq_16) {
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_s16_vmul_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMUL__WASMSIMD_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__WASMSIMD_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__WASMSIMD_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__WASMSIMD_U16, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__WASMSIMD_U16, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__WASMSIMD_U16, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_VMUL__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_s16_vmul_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMUL__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__NEON_U8, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_s16_vmul_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__NEON_U8, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__NEON_U8, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_VMUL__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_s16_vmul_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMUL__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__NEON_U16, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_s16_vmul_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__NEON_U16, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__NEON_U16, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
