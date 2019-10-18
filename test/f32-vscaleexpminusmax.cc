// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vscaleexpminusmax.h>
#include "vscaleexpminusmax-microkernel-tester.h"


#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_UNROLL64, n_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t n = 1; n < 64; n++) {
      VScaleExpMinusMaxMicrokernelTester()
        .n(n)
        .test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_unroll64);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_UNROLL64, n_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .n(64)
      .test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_unroll64);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_UNROLL64, n_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t n = 128; n < 384; n += 64) {
      VScaleExpMinusMaxMicrokernelTester()
        .n(n)
        .test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_unroll64);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_UNROLL64, n_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t n = 64; n < 128; n++) {
      VScaleExpMinusMaxMicrokernelTester()
        .n(n)
        .test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_unroll64);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_UNROLL64, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t n = 1; n < 100; n += 17) {
      VScaleExpMinusMaxMicrokernelTester()
        .n(n)
        .scale(0.01f)
        .test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_unroll64);
      VScaleExpMinusMaxMicrokernelTester()
        .n(n)
        .scale(100.0f)
        .test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_unroll64);
    }
  }
#endif  // CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_UNROLL128, n_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 1; n < 128; n++) {
      VScaleExpMinusMaxMicrokernelTester()
        .n(n)
        .test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_unroll128);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_UNROLL128, n_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .n(128)
      .test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_unroll128);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_UNROLL128, n_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 256; n < 768; n += 128) {
      VScaleExpMinusMaxMicrokernelTester()
        .n(n)
        .test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_unroll128);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_UNROLL128, n_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 128; n < 256; n++) {
      VScaleExpMinusMaxMicrokernelTester()
        .n(n)
        .test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_unroll128);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_UNROLL128, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t n = 1; n < 200; n += 35) {
      VScaleExpMinusMaxMicrokernelTester()
        .n(n)
        .scale(0.01f)
        .test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_unroll128);
      VScaleExpMinusMaxMicrokernelTester()
        .n(n)
        .scale(100.0f)
        .test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_unroll128);
    }
  }
#endif  // CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
