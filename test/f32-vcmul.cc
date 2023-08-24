// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vcmul.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/vbinary.h>
#include "vcmul-microkernel-tester.h"


TEST(F32_VCMUL__SCALAR_U1, batch_eq_1) {
  VCMulMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vcmul_ukernel__scalar_u1);
}

TEST(F32_VCMUL__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vcmul_ukernel__scalar_u1);
  }
}

TEST(F32_VCMUL__SCALAR_U1, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u1);
  }
}

TEST(F32_VCMUL__SCALAR_U1, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u1);
  }
}

TEST(F32_VCMUL__SCALAR_U1, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u1);
  }
}


TEST(F32_VCMUL__SCALAR_U2, batch_eq_2) {
  VCMulMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vcmul_ukernel__scalar_u2);
}

TEST(F32_VCMUL__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vcmul_ukernel__scalar_u2);
  }
}

TEST(F32_VCMUL__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vcmul_ukernel__scalar_u2);
  }
}

TEST(F32_VCMUL__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vcmul_ukernel__scalar_u2);
  }
}

TEST(F32_VCMUL__SCALAR_U2, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u2);
  }
}

TEST(F32_VCMUL__SCALAR_U2, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u2);
  }
}

TEST(F32_VCMUL__SCALAR_U2, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u2);
  }
}


TEST(F32_VCMUL__SCALAR_U4, batch_eq_4) {
  VCMulMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vcmul_ukernel__scalar_u4);
}

TEST(F32_VCMUL__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vcmul_ukernel__scalar_u4);
  }
}

TEST(F32_VCMUL__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vcmul_ukernel__scalar_u4);
  }
}

TEST(F32_VCMUL__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vcmul_ukernel__scalar_u4);
  }
}

TEST(F32_VCMUL__SCALAR_U4, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u4);
  }
}

TEST(F32_VCMUL__SCALAR_U4, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u4);
  }
}

TEST(F32_VCMUL__SCALAR_U4, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u4);
  }
}


TEST(F32_VCMUL__SCALAR_U8, batch_eq_8) {
  VCMulMicrokernelTester()
    .batch_size(8)
    .Test(xnn_f32_vcmul_ukernel__scalar_u8);
}

TEST(F32_VCMUL__SCALAR_U8, batch_div_8) {
  for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vcmul_ukernel__scalar_u8);
  }
}

TEST(F32_VCMUL__SCALAR_U8, batch_lt_8) {
  for (size_t batch_size = 1; batch_size < 8; batch_size++) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vcmul_ukernel__scalar_u8);
  }
}

TEST(F32_VCMUL__SCALAR_U8, batch_gt_8) {
  for (size_t batch_size = 9; batch_size < 16; batch_size++) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vcmul_ukernel__scalar_u8);
  }
}

TEST(F32_VCMUL__SCALAR_U8, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u8);
  }
}

TEST(F32_VCMUL__SCALAR_U8, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u8);
  }
}

TEST(F32_VCMUL__SCALAR_U8, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VCMulMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_f32_vcmul_ukernel__scalar_u8);
  }
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VCMUL__NEON_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VCMulMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vcmul_ukernel__neon_u4);
  }

  TEST(F32_VCMUL__NEON_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u4);
    }
  }

  TEST(F32_VCMUL__NEON_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u4);
    }
  }

  TEST(F32_VCMUL__NEON_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u4);
    }
  }

  TEST(F32_VCMUL__NEON_U4, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u4);
    }
  }

  TEST(F32_VCMUL__NEON_U4, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u4);
    }
  }

  TEST(F32_VCMUL__NEON_U4, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VCMUL__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VCMulMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vcmul_ukernel__neon_u8);
  }

  TEST(F32_VCMUL__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u8);
    }
  }

  TEST(F32_VCMUL__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u8);
    }
  }

  TEST(F32_VCMUL__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u8);
    }
  }

  TEST(F32_VCMUL__NEON_U8, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u8);
    }
  }

  TEST(F32_VCMUL__NEON_U8, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u8);
    }
  }

  TEST(F32_VCMUL__NEON_U8, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VCMUL__NEON_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VCMulMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vcmul_ukernel__neon_u12);
  }

  TEST(F32_VCMUL__NEON_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u12);
    }
  }

  TEST(F32_VCMUL__NEON_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u12);
    }
  }

  TEST(F32_VCMUL__NEON_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u12);
    }
  }

  TEST(F32_VCMUL__NEON_U12, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u12);
    }
  }

  TEST(F32_VCMUL__NEON_U12, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u12);
    }
  }

  TEST(F32_VCMUL__NEON_U12, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VCMUL__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VCMulMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vcmul_ukernel__neon_u16);
  }

  TEST(F32_VCMUL__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u16);
    }
  }

  TEST(F32_VCMUL__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u16);
    }
  }

  TEST(F32_VCMUL__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__neon_u16);
    }
  }

  TEST(F32_VCMUL__NEON_U16, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u16);
    }
  }

  TEST(F32_VCMUL__NEON_U16, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u16);
    }
  }

  TEST(F32_VCMUL__NEON_U16, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__neon_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VCMUL__SSE_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VCMulMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vcmul_ukernel__sse_u4);
  }

  TEST(F32_VCMUL__SSE_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u4);
    }
  }

  TEST(F32_VCMUL__SSE_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u4);
    }
  }

  TEST(F32_VCMUL__SSE_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u4);
    }
  }

  TEST(F32_VCMUL__SSE_U4, inplace_a) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u4);
    }
  }

  TEST(F32_VCMUL__SSE_U4, inplace_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u4);
    }
  }

  TEST(F32_VCMUL__SSE_U4, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VCMUL__SSE_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VCMulMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vcmul_ukernel__sse_u8);
  }

  TEST(F32_VCMUL__SSE_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u8);
    }
  }

  TEST(F32_VCMUL__SSE_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u8);
    }
  }

  TEST(F32_VCMUL__SSE_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u8);
    }
  }

  TEST(F32_VCMUL__SSE_U8, inplace_a) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u8);
    }
  }

  TEST(F32_VCMUL__SSE_U8, inplace_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u8);
    }
  }

  TEST(F32_VCMUL__SSE_U8, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VCMUL__SSE_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE;
    VCMulMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vcmul_ukernel__sse_u12);
  }

  TEST(F32_VCMUL__SSE_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u12);
    }
  }

  TEST(F32_VCMUL__SSE_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u12);
    }
  }

  TEST(F32_VCMUL__SSE_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u12);
    }
  }

  TEST(F32_VCMUL__SSE_U12, inplace_a) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u12);
    }
  }

  TEST(F32_VCMUL__SSE_U12, inplace_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u12);
    }
  }

  TEST(F32_VCMUL__SSE_U12, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u12);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VCMUL__SSE_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE;
    VCMulMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vcmul_ukernel__sse_u16);
  }

  TEST(F32_VCMUL__SSE_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u16);
    }
  }

  TEST(F32_VCMUL__SSE_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u16);
    }
  }

  TEST(F32_VCMUL__SSE_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__sse_u16);
    }
  }

  TEST(F32_VCMUL__SSE_U16, inplace_a) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u16);
    }
  }

  TEST(F32_VCMUL__SSE_U16, inplace_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u16);
    }
  }

  TEST(F32_VCMUL__SSE_U16, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__sse_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCMUL__WASMSIMD_U4, batch_eq_4) {
    VCMulMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vcmul_ukernel__wasmsimd_u4);
  }

  TEST(F32_VCMUL__WASMSIMD_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U4, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U4, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U4, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u4);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCMUL__WASMSIMD_U8, batch_eq_8) {
    VCMulMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vcmul_ukernel__wasmsimd_u8);
  }

  TEST(F32_VCMUL__WASMSIMD_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U8, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U8, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U8, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u8);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCMUL__WASMSIMD_U12, batch_eq_12) {
    VCMulMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vcmul_ukernel__wasmsimd_u12);
  }

  TEST(F32_VCMUL__WASMSIMD_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u12);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u12);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u12);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U12, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u12);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U12, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u12);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U12, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u12);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCMUL__WASMSIMD_U16, batch_eq_16) {
    VCMulMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vcmul_ukernel__wasmsimd_u16);
  }

  TEST(F32_VCMUL__WASMSIMD_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u16);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u16);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u16);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U16, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u16);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U16, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u16);
    }
  }

  TEST(F32_VCMUL__WASMSIMD_U16, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vcmul_ukernel__wasmsimd_u16);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
