// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-vmul-minmax-fp32.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/params-init.h>
#include <xnnpack/vmul.h>
#include "vmul-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VMulMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X8, inplace_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X8, inplace_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X8, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X8, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X8, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VMulMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X16, inplace_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X16, inplace_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X16, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X16, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE2_MUL16_LD64_X16, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VMulMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X8, inplace_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X8, inplace_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X8, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X8, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X8, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VMulMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X16, inplace_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X16, inplace_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X16, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X16, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__SSE41_MUL16_LD64_X16, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VMulMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X8, inplace_a) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X8, inplace_b) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X8, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X8, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X8, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VMulMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X16, inplace_a) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X16, inplace_b) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X16, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X16, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMUL_MINMAX_FP32__AVX_MUL16_LD64_X16, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
