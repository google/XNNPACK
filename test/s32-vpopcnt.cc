// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s32-vpopcnt.yaml
//   Generator: tools/generate-vunary-test.py


#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"
#include "next_prime.h"
#include "vunary-microkernel-tester.h"


TEST(S32_VPOPCNT__SCALAR_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_s32_vpopcnt_ukernel__scalar_u1);
}

TEST(S32_VPOPCNT__SCALAR_U1, batch_gt_1) {
  const size_t batch_step = 1;
  for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u1);
  }
}

TEST(S32_VPOPCNT__SCALAR_U1, inplace) {
  const size_t batch_step = 1;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u1);
  }
}


TEST(S32_VPOPCNT__SCALAR_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_s32_vpopcnt_ukernel__scalar_u2);
}

TEST(S32_VPOPCNT__SCALAR_U2, batch_div_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u2);
  }
}

TEST(S32_VPOPCNT__SCALAR_U2, batch_lt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u2);
  }
}

TEST(S32_VPOPCNT__SCALAR_U2, batch_gt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u2);
  }
}

TEST(S32_VPOPCNT__SCALAR_U2, inplace) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u2);
  }
}


TEST(S32_VPOPCNT__SCALAR_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_s32_vpopcnt_ukernel__scalar_u4);
}

TEST(S32_VPOPCNT__SCALAR_U4, batch_div_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u4);
  }
}

TEST(S32_VPOPCNT__SCALAR_U4, batch_lt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u4);
  }
}

TEST(S32_VPOPCNT__SCALAR_U4, batch_gt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u4);
  }
}

TEST(S32_VPOPCNT__SCALAR_U4, inplace) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u4);
  }
}


TEST(S32_VPOPCNT__SCALAR_U8, batch_eq_8) {
  VUnaryMicrokernelTester()
    .batch_size(8)
    .Test(xnn_s32_vpopcnt_ukernel__scalar_u8);
}

TEST(S32_VPOPCNT__SCALAR_U8, batch_div_8) {
  const size_t batch_step = 8;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u8);
  }
}

TEST(S32_VPOPCNT__SCALAR_U8, batch_lt_8) {
  const size_t batch_step = 8;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u8);
  }
}

TEST(S32_VPOPCNT__SCALAR_U8, batch_gt_8) {
  const size_t batch_step = 8;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u8);
  }
}

TEST(S32_VPOPCNT__SCALAR_U8, inplace) {
  const size_t batch_step = 8;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_s32_vpopcnt_ukernel__scalar_u8);
  }
}


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__SSE41_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_s32_vpopcnt_ukernel__sse41_u4);
  }

  TEST(S32_VPOPCNT__SSE41_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u4);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u4);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u4);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__SSE41_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_s32_vpopcnt_ukernel__sse41_u8);
  }

  TEST(S32_VPOPCNT__SSE41_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u8);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u8);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u8);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__SSE41_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_s32_vpopcnt_ukernel__sse41_u12);
  }

  TEST(S32_VPOPCNT__SSE41_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u12);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u12);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u12);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u12);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__SSE41_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_s32_vpopcnt_ukernel__sse41_u16);
  }

  TEST(S32_VPOPCNT__SSE41_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u16);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u16);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u16);
    }
  }

  TEST(S32_VPOPCNT__SSE41_U16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__sse41_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__AVX2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_s32_vpopcnt_ukernel__avx2_u8);
  }

  TEST(S32_VPOPCNT__AVX2_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u8);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u8);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u8);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__AVX2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_s32_vpopcnt_ukernel__avx2_u16);
  }

  TEST(S32_VPOPCNT__AVX2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u16);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u16);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u16);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__AVX2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_s32_vpopcnt_ukernel__avx2_u24);
  }

  TEST(S32_VPOPCNT__AVX2_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u24);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u24);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u24);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__AVX2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_s32_vpopcnt_ukernel__avx2_u32);
  }

  TEST(S32_VPOPCNT__AVX2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u32);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u32);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u32);
    }
  }

  TEST(S32_VPOPCNT__AVX2_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__avx2_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__AVX512F_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_s32_vpopcnt_ukernel__avx512f_u16);
  }

  TEST(S32_VPOPCNT__AVX512F_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u16);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u16);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u16);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__AVX512F_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_s32_vpopcnt_ukernel__avx512f_u32);
  }

  TEST(S32_VPOPCNT__AVX512F_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u32);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u32);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u32);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__AVX512F_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_s32_vpopcnt_ukernel__avx512f_u48);
  }

  TEST(S32_VPOPCNT__AVX512F_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u48);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u48);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u48);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_VPOPCNT__AVX512F_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_s32_vpopcnt_ukernel__avx512f_u64);
  }

  TEST(S32_VPOPCNT__AVX512F_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u64);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u64);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u64);
    }
  }

  TEST(S32_VPOPCNT__AVX512F_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__avx512f_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S32_VPOPCNT__WASMSIMD_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u4);
  }

  TEST(S32_VPOPCNT__WASMSIMD_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u4);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u4);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u4);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u4);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S32_VPOPCNT__WASMSIMD_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u8);
  }

  TEST(S32_VPOPCNT__WASMSIMD_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u8);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u8);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u8);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u8);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S32_VPOPCNT__WASMSIMD_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u12);
  }

  TEST(S32_VPOPCNT__WASMSIMD_U12, batch_div_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u12);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U12, batch_lt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u12);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U12, batch_gt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u12);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U12, inplace) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u12);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S32_VPOPCNT__WASMSIMD_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u16);
  }

  TEST(S32_VPOPCNT__WASMSIMD_U16, batch_div_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u16);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U16, batch_lt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u16);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U16, batch_gt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u16);
    }
  }

  TEST(S32_VPOPCNT__WASMSIMD_U16, inplace) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__wasmsimd_u16);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S32_VPOPCNT__NEON_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_s32_vpopcnt_ukernel__neon_u4);
  }

  TEST(S32_VPOPCNT__NEON_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u4);
    }
  }

  TEST(S32_VPOPCNT__NEON_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u4);
    }
  }

  TEST(S32_VPOPCNT__NEON_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u4);
    }
  }

  TEST(S32_VPOPCNT__NEON_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S32_VPOPCNT__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_s32_vpopcnt_ukernel__neon_u8);
  }

  TEST(S32_VPOPCNT__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u8);
    }
  }

  TEST(S32_VPOPCNT__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u8);
    }
  }

  TEST(S32_VPOPCNT__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u8);
    }
  }

  TEST(S32_VPOPCNT__NEON_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S32_VPOPCNT__NEON_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_s32_vpopcnt_ukernel__neon_u12);
  }

  TEST(S32_VPOPCNT__NEON_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u12);
    }
  }

  TEST(S32_VPOPCNT__NEON_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u12);
    }
  }

  TEST(S32_VPOPCNT__NEON_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u12);
    }
  }

  TEST(S32_VPOPCNT__NEON_U12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S32_VPOPCNT__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_s32_vpopcnt_ukernel__neon_u16);
  }

  TEST(S32_VPOPCNT__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u16);
    }
  }

  TEST(S32_VPOPCNT__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u16);
    }
  }

  TEST(S32_VPOPCNT__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u16);
    }
  }

  TEST(S32_VPOPCNT__NEON_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s32_vpopcnt_ukernel__neon_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
