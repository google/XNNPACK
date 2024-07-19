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


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S16_VMUL__AVX2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_s16_vmul_ukernel__avx2_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMUL__AVX2_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U8, inplace_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U8, inplace_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U8, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
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
  TEST(S16_VMUL__AVX2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VBinaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_s16_vmul_ukernel__avx2_u24, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMUL__AVX2_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u24, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u24, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmul_ukernel__avx2_u24, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U24, inplace_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u24, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U24, inplace_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u24, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMUL__AVX2_U24, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_s16_vmul_ukernel__avx2_u24, VBinaryMicrokernelTester::OpType::Mul, xnn_init_s16_cvt_scalar_params);
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
