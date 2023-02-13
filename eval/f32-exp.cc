// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits>

#include <gtest/gtest.h>

#include "math-evaluation-tester.h"

#include <xnnpack/isa-checks.h>


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXP__NEONFMA_RR2_LUT64_P2, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__neonfma_rr2_lut64_p2, 1.0f);
  }

  TEST(EXP__NEONFMA_RR2_LUT64_P2, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__neonfma_rr2_lut64_p2, 1.0f);
  }

  TEST(EXP__NEONFMA_RR2_LUT64_P2, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__neonfma_rr2_lut64_p2, 0.0f);
  }

  TEST(EXP__NEONFMA_RR2_LUT64_P2, positive_overflow) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__neonfma_rr2_lut64_p2, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__NEONFMA_RR2_LUT64_P2, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__neonfma_rr2_lut64_p2);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXP__NEONFMA_RR2_P5, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__neonfma_rr2_p5, 1.0f);
  }

  TEST(EXP__NEONFMA_RR2_P5, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__neonfma_rr2_p5, 1.0f);
  }

  TEST(EXP__NEONFMA_RR2_P5, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__neonfma_rr2_p5, 0.0f);
  }

  TEST(EXP__NEONFMA_RR2_P5, positive_overflow) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__neonfma_rr2_p5, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__NEONFMA_RR2_P5, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__neonfma_rr2_p5);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__AVX512F_RR2_LUT16_P3_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT16_P3_PERM, positive_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT16_P3_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm, 0.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT16_P3_PERM, positive_overflow) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__AVX512F_RR2_LUT16_P3_PERM, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__AVX512F_RR2_LUT16_P3_PERM_SCALEF, negative_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm_scalef, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT16_P3_PERM_SCALEF, positive_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm_scalef, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT16_P3_PERM_SCALEF, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm_scalef, 0.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT16_P3_PERM_SCALEF, positive_overflow) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm_scalef, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__AVX512F_RR2_LUT16_P3_PERM_SCALEF, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm_scalef);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__AVX512F_RR2_LUT32_P2_PERM2, negative_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT32_P2_PERM2, positive_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT32_P2_PERM2, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2, 0.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT32_P2_PERM2, positive_overflow) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__AVX512F_RR2_LUT32_P2_PERM2, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__AVX512F_RR2_LUT32_P2_PERM2_SCALEF, negative_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2_scalef, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT32_P2_PERM2_SCALEF, positive_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2_scalef, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT32_P2_PERM2_SCALEF, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2_scalef, 0.0f);
  }

  TEST(EXP__AVX512F_RR2_LUT32_P2_PERM2_SCALEF, positive_overflow) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2_scalef, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__AVX512F_RR2_LUT32_P2_PERM2_SCALEF, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2_scalef);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__AVX512F_RR2_P5, negative_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_p5, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_P5, positive_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_p5, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_P5, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_p5, 0.0f);
  }

  TEST(EXP__AVX512F_RR2_P5, positive_overflow) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_p5, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__AVX512F_RR2_P5, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__avx512f_rr2_p5);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__AVX512F_RR2_P5_SCALEF, negative_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_p5_scalef, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_P5_SCALEF, positive_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_p5_scalef, 1.0f);
  }

  TEST(EXP__AVX512F_RR2_P5_SCALEF, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_p5_scalef, 0.0f);
  }

  TEST(EXP__AVX512F_RR2_P5_SCALEF, positive_overflow) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__avx512f_rr2_p5_scalef, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__AVX512F_RR2_P5_SCALEF, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__avx512f_rr2_p5_scalef);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__AVX2_RR2_LUT8_P3_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_lut8_p3_perm, 1.0f);
  }

  TEST(EXP__AVX2_RR2_LUT8_P3_PERM, positive_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_lut8_p3_perm, 1.0f);
  }

  TEST(EXP__AVX2_RR2_LUT8_P3_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_lut8_p3_perm, 0.0f);
  }

  TEST(EXP__AVX2_RR2_LUT8_P3_PERM, positive_overflow) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_lut8_p3_perm, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__AVX2_RR2_LUT8_P3_PERM, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__avx2_rr2_lut8_p3_perm);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__AVX2_RR2_LUT8_P4_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_lut8_p4_perm, 1.0f);
  }

  TEST(EXP__AVX2_RR2_LUT8_P4_PERM, positive_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_lut8_p4_perm, 1.0f);
  }

  TEST(EXP__AVX2_RR2_LUT8_P4_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_lut8_p4_perm, 0.0f);
  }

  TEST(EXP__AVX2_RR2_LUT8_P4_PERM, positive_overflow) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_lut8_p4_perm, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__AVX2_RR2_LUT8_P4_PERM, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__avx2_rr2_lut8_p4_perm);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__AVX2_RR2_P5, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_p5, 1.0f);
  }

  TEST(EXP__AVX2_RR2_P5, positive_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_p5, 1.0f);
  }

  TEST(EXP__AVX2_RR2_P5, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_p5, 0.0f);
  }

  TEST(EXP__AVX2_RR2_P5, positive_overflow) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__avx2_rr2_p5, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__AVX2_RR2_P5, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__avx2_rr2_p5);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__AVX_RR2_P5, negative_zero) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx_rr2_p5, 1.0f);
  }

  TEST(EXP__AVX_RR2_P5, positive_zero) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx_rr2_p5, 1.0f);
  }

  TEST(EXP__AVX_RR2_P5, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__avx_rr2_p5, 0.0f);
  }

  TEST(EXP__AVX_RR2_P5, positive_overflow) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__avx_rr2_p5, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__AVX_RR2_P5, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__avx_rr2_p5);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__SSE2_RR2_LUT64_P2, negative_zero) {
    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__sse2_rr2_lut64_p2, 1.0f);
  }

  TEST(EXP__SSE2_RR2_LUT64_P2, positive_zero) {
    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__sse2_rr2_lut64_p2, 1.0f);
  }

  TEST(EXP__SSE2_RR2_LUT64_P2, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__sse2_rr2_lut64_p2, 0.0f);
  }

  TEST(EXP__SSE2_RR2_LUT64_P2, positive_overflow) {
    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__sse2_rr2_lut64_p2, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__SSE2_RR2_LUT64_P2, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__sse2_rr2_lut64_p2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXP__SSE2_RR2_P5, negative_zero) {
    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__sse2_rr2_p5, 1.0f);
  }

  TEST(EXP__SSE2_RR2_P5, positive_zero) {
    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_exp__sse2_rr2_p5, 1.0f);
  }

  TEST(EXP__SSE2_RR2_P5, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.9FE36Ap+6f)
      .TestOutputMatchReference(xnn_math_f32_exp__sse2_rr2_p5, 0.0f);
  }

  TEST(EXP__SSE2_RR2_P5, positive_overflow) {
    MathEvaluationTester()
      .input_range(0x1.62E430p+6f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_exp__sse2_rr2_p5, std::numeric_limits<float>::infinity());
  }

  TEST(EXP__SSE2_RR2_P5, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_exp__sse2_rr2_p5);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
