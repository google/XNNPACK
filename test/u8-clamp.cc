// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/u8-clamp.yaml
//   Generator: tools/generate-clamp-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/clamp.h>
#include "clamp-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U8_CLAMP__NEON_X64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    ClampMicrokernelTester()
      .batch_size(64)
      .Test(xnn_u8_clamp_ukernel__neon_x64);
  }

  TEST(U8_CLAMP__NEON_X64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_clamp_ukernel__neon_x64);
    }
  }

  TEST(U8_CLAMP__NEON_X64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_clamp_ukernel__neon_x64);
    }
  }

  TEST(U8_CLAMP__NEON_X64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_clamp_ukernel__neon_x64);
    }
  }

  TEST(U8_CLAMP__NEON_X64, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_u8_clamp_ukernel__neon_x64);
    }
  }

  TEST(U8_CLAMP__NEON_X64, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_u8_clamp_ukernel__neon_x64);
      }
    }
  }

  TEST(U8_CLAMP__NEON_X64, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .batch_size(batch_size)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_u8_clamp_ukernel__neon_x64);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(U8_CLAMP__SSE2_X64, batch_eq_64) {
    TEST_REQUIRES_X86_SSE2;
    ClampMicrokernelTester()
      .batch_size(64)
      .Test(xnn_u8_clamp_ukernel__sse2_x64);
  }

  TEST(U8_CLAMP__SSE2_X64, batch_div_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_clamp_ukernel__sse2_x64);
    }
  }

  TEST(U8_CLAMP__SSE2_X64, batch_lt_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_clamp_ukernel__sse2_x64);
    }
  }

  TEST(U8_CLAMP__SSE2_X64, batch_gt_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_clamp_ukernel__sse2_x64);
    }
  }

  TEST(U8_CLAMP__SSE2_X64, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_u8_clamp_ukernel__sse2_x64);
    }
  }

  TEST(U8_CLAMP__SSE2_X64, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_u8_clamp_ukernel__sse2_x64);
      }
    }
  }

  TEST(U8_CLAMP__SSE2_X64, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .batch_size(batch_size)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_u8_clamp_ukernel__sse2_x64);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(U8_CLAMP__SCALAR_X4, batch_eq_4) {
  ClampMicrokernelTester()
    .batch_size(4)
    .Test(xnn_u8_clamp_ukernel__scalar_x4, ClampMicrokernelTester::Variant::Scalar);
}

TEST(U8_CLAMP__SCALAR_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    ClampMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_clamp_ukernel__scalar_x4, ClampMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_CLAMP__SCALAR_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    ClampMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_clamp_ukernel__scalar_x4, ClampMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_CLAMP__SCALAR_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    ClampMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_clamp_ukernel__scalar_x4, ClampMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_CLAMP__SCALAR_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    ClampMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_u8_clamp_ukernel__scalar_x4, ClampMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_CLAMP__SCALAR_X4, qmin) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .qmin(qmin)
        .qmax(255)
        .Test(xnn_u8_clamp_ukernel__scalar_x4, ClampMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(U8_CLAMP__SCALAR_X4, qmax) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .qmin(0)
        .qmax(qmax)
        .Test(xnn_u8_clamp_ukernel__scalar_x4, ClampMicrokernelTester::Variant::Scalar);
    }
  }
}