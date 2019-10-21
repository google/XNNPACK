// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vscale.h>
#include "vscale-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALE__AVX_UNROLL32, n_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VScaleMicrokernelTester()
      .n(32)
      .Test(xnn_f32_vscale_ukernel__avx_unroll32);
  }

  TEST(F32_VSCALE__AVX_UNROLL32, n_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 64; n < 256; n += 32) {
      VScaleMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vscale_ukernel__avx_unroll32);
    }
  }

  TEST(F32_VSCALE__AVX_UNROLL32, n_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 1; n < 32; n++) {
      VScaleMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vscale_ukernel__avx_unroll32);
    }
  }

  TEST(F32_VSCALE__AVX_UNROLL32, n_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 33; n < 64; n++) {
      VScaleMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vscale_ukernel__avx_unroll32);
    }
  }

  TEST(F32_VSCALE__AVX_UNROLL32, inplace_n_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VScaleMicrokernelTester()
      .n(32)
      .inplace(true)
      .Test(xnn_f32_vscale_ukernel__avx_unroll32);
  }

  TEST(F32_VSCALE__AVX_UNROLL32, inplace_n_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 64; n < 256; n += 32) {
      VScaleMicrokernelTester()
        .n(n)
        .inplace(true)
        .Test(xnn_f32_vscale_ukernel__avx_unroll32);
    }
  }

  TEST(F32_VSCALE__AVX_UNROLL32, inplace_n_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 1; n < 32; n++) {
      VScaleMicrokernelTester()
        .n(n)
        .inplace(true)
        .Test(xnn_f32_vscale_ukernel__avx_unroll32);
    }
  }

  TEST(F32_VSCALE__AVX_UNROLL32, inplace_n_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t n = 33; n < 64; n++) {
      VScaleMicrokernelTester()
        .n(n)
        .inplace(true)
        .Test(xnn_f32_vscale_ukernel__avx_unroll32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F64_VSCALE__AVX512F_UNROLL64, n_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleMicrokernelTester()
      .n(64)
      .Test(xnn_f32_vscale_ukernel__avx512f_unroll64);
  }

  TEST(F64_VSCALE__AVX512F_UNROLL64, n_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 128; n < 512; n += 64) {
      VScaleMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vscale_ukernel__avx512f_unroll64);
    }
  }

  TEST(F64_VSCALE__AVX512F_UNROLL64, n_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 1; n < 64; n++) {
      VScaleMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vscale_ukernel__avx512f_unroll64);
    }
  }

  TEST(F64_VSCALE__AVX512F_UNROLL64, n_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 65; n < 128; n++) {
      VScaleMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vscale_ukernel__avx512f_unroll64);
    }
  }

  TEST(F64_VSCALE__AVX512F_UNROLL64, inplace_n_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleMicrokernelTester()
      .n(64)
      .inplace(true)
      .Test(xnn_f32_vscale_ukernel__avx512f_unroll64);
  }

  TEST(F64_VSCALE__AVX512F_UNROLL64, inplace_n_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 64; n < 512; n += 64) {
      VScaleMicrokernelTester()
        .n(n)
        .inplace(true)
        .Test(xnn_f32_vscale_ukernel__avx512f_unroll64);
    }
  }

  TEST(F64_VSCALE__AVX512F_UNROLL64, inplace_n_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 1; n < 64; n++) {
      VScaleMicrokernelTester()
        .n(n)
        .inplace(true)
        .Test(xnn_f32_vscale_ukernel__avx512f_unroll64);
    }
  }

  TEST(F64_VSCALE__AVX512F_UNROLL64, inplace_n_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 65; n < 128; n++) {
      VScaleMicrokernelTester()
        .n(n)
        .inplace(true)
        .Test(xnn_f32_vscale_ukernel__avx512f_unroll64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
