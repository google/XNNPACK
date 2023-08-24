// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-raddextexp.yaml
//   Generator: tools/generate-raddextexp-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/raddextexp.h>
#include "raddextexp-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U64, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U64, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U64, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U64, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U64_ACC2, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc2);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U64_ACC2, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U64_ACC2, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U64_ACC2, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U64_ACC4, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc4);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U64_ACC4, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc4);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U64_ACC4, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc4);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U64_ACC4, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U72, elements_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(72)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u72);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U72, elements_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 144; elements < 720; elements += 72) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u72);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U72, elements_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 72; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u72);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U72, elements_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 73; elements < 144; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u72);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U72_ACC3, elements_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(72)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u72_acc3);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U72_ACC3, elements_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 144; elements < 720; elements += 72) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u72_acc3);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U72_ACC3, elements_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 72; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u72_acc3);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U72_ACC3, elements_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 73; elements < 144; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u72_acc3);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U80, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U80, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U80, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U80, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U80_ACC2, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc2);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U80_ACC2, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U80_ACC2, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U80_ACC2, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U80_ACC5, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc5);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U80_ACC5, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc5);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U80_ACC5, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc5);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U80_ACC5, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc5);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U96, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC2, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc2);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC2, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC2, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC2, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC3, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc3);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC3, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc3);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC3, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc3);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC3, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc3);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC6, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddExtExpMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc6);
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC6, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc6);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC6, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc6);
    }
  }

  TEST(F32_RADDEXTEXP__AVX2_P5_U96_ACC6, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc6);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128, elements_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128, elements_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128, elements_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128, elements_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128_ACC2, elements_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc2);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128_ACC2, elements_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128_ACC2, elements_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128_ACC2, elements_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128_ACC4, elements_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc4);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128_ACC4, elements_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc4);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128_ACC4, elements_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc4);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U128_ACC4, elements_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U144, elements_eq_144) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(144)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U144, elements_div_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 288; elements < 1440; elements += 144) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U144, elements_lt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 144; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U144, elements_gt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 145; elements < 288; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U144_ACC3, elements_eq_144) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(144)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144_acc3);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U144_ACC3, elements_div_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 288; elements < 1440; elements += 144) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144_acc3);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U144_ACC3, elements_lt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 144; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144_acc3);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U144_ACC3, elements_gt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 145; elements < 288; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144_acc3);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160, elements_eq_160) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(160)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160, elements_div_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 320; elements < 1600; elements += 160) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160, elements_lt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 160; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160, elements_gt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 161; elements < 320; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160_ACC2, elements_eq_160) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(160)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc2);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160_ACC2, elements_div_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 320; elements < 1600; elements += 160) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160_ACC2, elements_lt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 160; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160_ACC2, elements_gt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 161; elements < 320; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160_ACC5, elements_eq_160) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(160)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc5);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160_ACC5, elements_div_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 320; elements < 1600; elements += 160) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc5);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160_ACC5, elements_lt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 160; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc5);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U160_ACC5, elements_gt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 161; elements < 320; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc5);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC2, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc2);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC2, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC2, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc2);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC2, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC3, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc3);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC3, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc3);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC3, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc3);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC3, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc3);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC6, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddExtExpMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc6);
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC6, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc6);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC6, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc6);
    }
  }

  TEST(F32_RADDEXTEXP__AVX512F_P5_SCALEF_U192_ACC6, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddExtExpMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc6);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
