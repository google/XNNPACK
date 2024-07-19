// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s16-vmulc.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"
#include "vbinaryc-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S16_VMULC__AVX2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_s16_vmulc_ukernel__avx2_u8, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMULC__AVX2_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u8, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u8, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u8, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s16_vmulc_ukernel__avx2_u8, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S16_VMULC__AVX2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_s16_vmulc_ukernel__avx2_u16, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMULC__AVX2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u16, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u16, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u16, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s16_vmulc_ukernel__avx2_u16, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S16_VMULC__AVX2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VBinaryCMicrokernelTester()
      .batch_size(24)
      .Test(xnn_s16_vmulc_ukernel__avx2_u24, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMULC__AVX2_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u24, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u24, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u24, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s16_vmulc_ukernel__avx2_u24, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S16_VMULC__AVX2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VBinaryCMicrokernelTester()
      .batch_size(32)
      .Test(xnn_s16_vmulc_ukernel__avx2_u32, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
  }

  TEST(S16_VMULC__AVX2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u32, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u32, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s16_vmulc_ukernel__avx2_u32, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }

  TEST(S16_VMULC__AVX2_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s16_vmulc_ukernel__avx2_u32, VBinaryCMicrokernelTester::OpType::MulC, xnn_init_s16_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
