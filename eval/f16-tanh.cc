// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits>

#include <gtest/gtest.h>

#include "math-evaluation-tester.h"

#include <xnnpack/isa-checks.h>


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1_RR1_P3_DIV, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__aarch64_neonfp16arith_expm1_rr1_p3_div, 1.0f);
  }

  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1_RR1_P3_DIV, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__aarch64_neonfp16arith_expm1_rr1_p3_div, -1.0f);
  }

  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1_RR1_P3_DIV, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__aarch64_neonfp16arith_expm1_rr1_p3_div);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1fma, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1fma, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1fma);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1recps, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1recps, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_NR1RECPS, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_nr1recps);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_RECPE, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_recpe, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_RECPE, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_recpe, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1_RR1_P3_RECPE, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;

    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1_rr1_p3_recpe);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1_RR1_P3_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__avx2_expm1_rr1_p3_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1_RR1_P3_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__avx2_expm1_rr1_p3_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1_RR1_P3_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__avx2_expm1_rr1_p3_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1_RR1_P3_RCP, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__avx2_expm1_rr1_p3_rcp, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1_RR1_P3_RCP, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__avx2_expm1_rr1_p3_rcp, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1_RR1_P3_RCP, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__avx2_expm1_rr1_p3_rcp);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_P17, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_p17, 1.0f);
  }

  TEST(TANH__FMA3_P17, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_p17, -1.0f);
  }

  TEST(TANH__FMA3_P17, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__fma3_p17);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_P19, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_p19, 1.0f);
  }

  TEST(TANH__FMA3_P19, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_p19, -1.0f);
  }

  TEST(TANH__FMA3_P19, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__fma3_p19);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__F16C_P19, positive_saturation) {
    TEST_REQUIRES_X86_F16C;

    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__f16c_p19, 1.0f);
  }

  TEST(TANH__F16C_P19, negative_saturation) {
    TEST_REQUIRES_X86_F16C;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__f16c_p19, -1.0f);
  }

  TEST(TANH__F16C_P19, nan) {
    TEST_REQUIRES_X86_F16C;

    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__f16c_p19);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
