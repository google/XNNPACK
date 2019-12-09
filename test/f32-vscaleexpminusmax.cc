// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vscaleexpminusmax.yaml
//   Generator: tools/generate-vscaleexpminusmax-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vscaleexpminusmax.h>
#include "vscaleexpminusmax-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X8, elements_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x8);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X8, elements_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 16; elements < 80; elements += 8) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x8);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X8, elements_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 8; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x8);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X8, elements_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 9; elements < 16; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x8);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X8, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 40; elements += 7) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x8);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X16, elements_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x16);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X16, elements_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x16);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X16, elements_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 16; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x16);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X16, elements_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 17; elements < 32; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x16);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X16, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 80; elements += 15) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x16);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X24, elements_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(24)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X24, elements_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 48; elements < 240; elements += 24) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X24, elements_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 24; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X24, elements_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 25; elements < 48; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X24, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 120; elements += 23) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X32, elements_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x32);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X32, elements_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 64; elements < 320; elements += 32) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x32);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X32, elements_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 32; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x32);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X32, elements_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 33; elements < 64; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x32);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X32, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 160; elements += 31) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x32);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X40, elements_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(40)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x40);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X40, elements_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 80; elements < 400; elements += 40) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x40);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X40, elements_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 40; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x40);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X40, elements_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 41; elements < 80; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x40);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X40, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 200; elements += 39) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x40);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x40);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X48, elements_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(48)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x48);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X48, elements_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 96; elements < 480; elements += 48) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x48);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X48, elements_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 48; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x48);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X48, elements_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 49; elements < 96; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x48);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X48, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 240; elements += 47) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x48);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X56, elements_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(56)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x56);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X56, elements_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 112; elements < 560; elements += 56) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x56);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X56, elements_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 56; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x56);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X56, elements_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 57; elements < 112; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x56);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X56, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 280; elements += 55) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x56);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x56);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X64, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x64);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X64, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x64);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X64, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x64);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X64, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x64);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X64, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 320; elements += 63) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x64);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X72, elements_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(72)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x72);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X72, elements_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 144; elements < 720; elements += 72) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x72);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X72, elements_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 72; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x72);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X72, elements_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 73; elements < 144; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x72);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X72, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 360; elements += 71) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x72);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x72);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X80, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x80);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X80, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x80);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X80, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x80);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X80, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x80);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X80, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 400; elements += 79) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x80);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X88, elements_eq_88) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(88)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x88);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X88, elements_div_88) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 176; elements < 880; elements += 88) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x88);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X88, elements_lt_88) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 88; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x88);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X88, elements_gt_88) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 89; elements < 176; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x88);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X88, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 440; elements += 87) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x88);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x88);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X96, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x96);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X96, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x96);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X96, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x96);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X96, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x96);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX2_P5_X96, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements <= 480; elements += 95) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x96);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X16, elements_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x16);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X16, elements_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 32; elements < 160; elements += 16) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x16);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X16, elements_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 16; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x16);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X16, elements_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 17; elements < 32; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x16);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X16, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 80; elements += 15) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x16);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X32, elements_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x32);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X32, elements_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 64; elements < 320; elements += 32) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x32);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X32, elements_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 32; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x32);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X32, elements_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 33; elements < 64; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x32);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X32, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 160; elements += 31) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x32);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X48, elements_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(48)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x48);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X48, elements_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 96; elements < 480; elements += 48) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x48);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X48, elements_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 48; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x48);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X48, elements_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 49; elements < 96; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x48);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X48, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 240; elements += 47) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x48);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X64, elements_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x64);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X64, elements_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 128; elements < 640; elements += 64) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x64);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X64, elements_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 64; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x64);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X64, elements_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 65; elements < 128; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x64);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X64, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 320; elements += 63) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x64);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X80, elements_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x80);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X80, elements_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 160; elements < 800; elements += 80) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x80);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X80, elements_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 80; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x80);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X80, elements_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 81; elements < 160; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x80);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X80, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 400; elements += 79) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x80);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X96, elements_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x96);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X96, elements_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 192; elements < 960; elements += 96) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x96);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X96, elements_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 96; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x96);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X96, elements_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 97; elements < 192; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x96);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X96, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 480; elements += 95) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x96);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X112, elements_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(112)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x112);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X112, elements_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 224; elements < 1120; elements += 112) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x112);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X112, elements_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 112; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x112);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X112, elements_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 113; elements < 224; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x112);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X112, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 560; elements += 111) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x112);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x112);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X128, elements_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x128);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X128, elements_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x128);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X128, elements_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 128; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x128);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X128, elements_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 129; elements < 256; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x128);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X128, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 640; elements += 127) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x128);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X144, elements_eq_144) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(144)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x144);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X144, elements_div_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 288; elements < 1440; elements += 144) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x144);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X144, elements_lt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 144; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x144);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X144, elements_gt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 145; elements < 288; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x144);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X144, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 720; elements += 143) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x144);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x144);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X160, elements_eq_160) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(160)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x160);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X160, elements_div_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 320; elements < 1600; elements += 160) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x160);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X160, elements_lt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 160; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x160);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X160, elements_gt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 161; elements < 320; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x160);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X160, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 800; elements += 159) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x160);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x160);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X176, elements_eq_176) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(176)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x176);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X176, elements_div_176) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 352; elements < 1760; elements += 176) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x176);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X176, elements_lt_176) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 176; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x176);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X176, elements_gt_176) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 177; elements < 352; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x176);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X176, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 880; elements += 175) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x176);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x176);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X192, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    VScaleExpMinusMaxMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x192);
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X192, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x192);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X192, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x192);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X192, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x192);
    }
  }

  TEST(F32_VSCALEEXPMINUSMAX__AVX512F_P5_SCALEF_X192, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements <= 960; elements += 191) {
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(0.01f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x192);
      VScaleExpMinusMaxMicrokernelTester()
        .elements(elements)
        .scale(100.0f)
        .Test(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x192);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
