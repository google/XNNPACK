// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-vadd-minmax.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vadd.h>
#include "vadd-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VAddMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x8);
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X8, inplace_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X8, inplace_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X8, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X8, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X8, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VAddMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x16);
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X16, inplace_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X16, inplace_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X16, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X16, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X16, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VAddMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x24);
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X24, inplace_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X24, inplace_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X24, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X24, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X24, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    VAddMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x32);
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X32, inplace_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X32, inplace_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X32, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X32, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE2_MUL16_LD64_X32, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VAddMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x8);
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X8, inplace_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X8, inplace_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X8, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X8, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X8, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VAddMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x16);
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X16, inplace_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X16, inplace_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X16, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X16, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X16, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VAddMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x24);
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X24, inplace_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X24, inplace_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X24, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X24, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X24, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    VAddMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x32);
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X32, inplace_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X32, inplace_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X32, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X32, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADD_MINMAX__SSE41_MUL16_LD64_X32, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
