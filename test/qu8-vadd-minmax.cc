// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-vadd-minmax.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vadd.h>
#include "vadd-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VADD_MINMAX__SSE2_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VAddMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vadd_minmax_ukernel__sse2_x8, xnn_init_qu8_add_minmax_params);
  }

  TEST(QU8_VADD_MINMAX__SSE2_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2_x8, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2_x8, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2_x8, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2_X8, inplace_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2_x8, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2_X8, inplace_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2_x8, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2_X8, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2_x8, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2_X8, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2_x8, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2_X8, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2_x8, xnn_init_qu8_add_minmax_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_VADD_MINMAX__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VAddMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qu8_vadd_minmax_ukernel__neon_x32, xnn_init_qu8_add_minmax_params);
  }

  TEST(QU8_VADD_MINMAX__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon_x32, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon_x32, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon_x32, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON_X32, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon_x32, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON_X32, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon_x32, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON_X32, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon_x32, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON_X32, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon_x32, xnn_init_qu8_add_minmax_params);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON_X32, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon_x32, xnn_init_qu8_add_minmax_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(QU8_VADD_MINMAX__SCALAR_X1, batch_eq_1) {
  VAddMicrokernelTester()
    .batch_size(1)
    .Test(xnn_qu8_vadd_minmax_ukernel__scalar_x1, xnn_init_qu8_add_minmax_scalar_params);
}

TEST(QU8_VADD_MINMAX__SCALAR_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VAddMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar_x1, xnn_init_qu8_add_minmax_scalar_params);
  }
}

TEST(QU8_VADD_MINMAX__SCALAR_X1, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VAddMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar_x1, xnn_init_qu8_add_minmax_scalar_params);
  }
}

TEST(QU8_VADD_MINMAX__SCALAR_X1, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VAddMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar_x1, xnn_init_qu8_add_minmax_scalar_params);
  }
}

TEST(QU8_VADD_MINMAX__SCALAR_X1, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VAddMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar_x1, xnn_init_qu8_add_minmax_scalar_params);
  }
}

TEST(QU8_VADD_MINMAX__SCALAR_X1, qmin) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VAddMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar_x1, xnn_init_qu8_add_minmax_scalar_params);
  }
}

TEST(QU8_VADD_MINMAX__SCALAR_X1, qmax) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VAddMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar_x1, xnn_init_qu8_add_minmax_scalar_params);
  }
}