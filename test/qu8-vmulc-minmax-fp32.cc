// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-vmulc-minmax-fp32.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/params-init.h>
#include <xnnpack/vmul.h>
#include "vmulc-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VMulCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X8, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X8, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VMulCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X16, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_X16, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VMulCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X8, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X8, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VMulCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X16, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_X16, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VMulCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X8, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X8, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x8, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VMulCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X16, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_X16, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_x16, xnn_init_qu8_mul_minmax_fp32_sse2_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X8, batch_eq_8) {
    VMulCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X8, qmin) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X8, qmax) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x8, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X16, batch_eq_16) {
    VMulCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x16, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x16, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x16, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x16, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x16, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X16, qmin) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x16, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_X16, qmax) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VMulCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x16, xnn_init_qu8_mul_minmax_fp32_wasmsimd_params, xnn_init_qu8_requantization_fp32_params, xnn_qu8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD
