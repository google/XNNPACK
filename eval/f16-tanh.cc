// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: eval/f16-tanh.yaml
//   Generator: tools/generate-tanh-eval.py


#include <limits>

#include <gtest/gtest.h>

#include "math-evaluation-tester.h"

#include <xnnpack/isa-checks.h>


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_DIV, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__aarch64_neonfp16arith_expm1minus_rr1_p3h1ts_div, 1.0f);
  }

  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_DIV, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__aarch64_neonfp16arith_expm1minus_rr1_p3h1ts_div, -1.0f);
  }

  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_DIV, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__aarch64_neonfp16arith_expm1minus_rr1_p3h1ts_div);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div, 1.0f);
  }

  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div, -1.0f);
  }

  TEST(TANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1fma, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1fma, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1fma);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1FMAADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1fmaadj, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1FMAADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1fmaadj, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1FMAADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1fmaadj);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1recps, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1recps, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1RECPS, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1recps);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1RECPSADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1recpsadj, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1RECPSADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1recpsadj, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_NR1RECPSADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1recpsadj);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_RECPE, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_recpe, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_RECPE, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_recpe, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_RECPE, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_recpe);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_RECPEADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_recpeadj, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_RECPEADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_recpeadj, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H1TS_RECPEADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_recpeadj);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMAADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fmaadj, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMAADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fmaadj, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMAADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fmaadj);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPSADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recpsadj, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPSADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recpsadj, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPSADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recpsadj);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPE, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_recpe, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPE, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_recpe, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPE, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_recpe);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj, 1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj, -1.0f);
  }

  TEST(TANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__f16c_expm1minus_rr1_p3h2ts_div, 1.0f);
  }

  TEST(TANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__f16c_expm1minus_rr1_p3h2ts_div, -1.0f);
  }

  TEST(TANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV, nan) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__f16c_expm1minus_rr1_p3h2ts_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP, positive_saturation) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__f16c_expm1minus_rr1_p3h2ts_rcp, 1.0f);
  }

  TEST(TANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP, negative_saturation) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__f16c_expm1minus_rr1_p3h2ts_rcp, -1.0f);
  }

  TEST(TANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP, nan) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__f16c_expm1minus_rr1_p3h2ts_rcp);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__F16C_POLYNOMIAL_P17H8T2, positive_saturation) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__f16c_polynomial_p17h8t2, 1.0f);
  }

  TEST(TANH__F16C_POLYNOMIAL_P17H8T2, negative_saturation) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__f16c_polynomial_p17h8t2, -1.0f);
  }

  TEST(TANH__F16C_POLYNOMIAL_P17H8T2, nan) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__f16c_polynomial_p17h8t2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__F16C_POLYNOMIAL_P19H9T2, positive_saturation) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__f16c_polynomial_p19h9t2, 1.0f);
  }

  TEST(TANH__F16C_POLYNOMIAL_P19H9T2, negative_saturation) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__f16c_polynomial_p19h9t2, -1.0f);
  }

  TEST(TANH__F16C_POLYNOMIAL_P19H9T2, nan) {
    TEST_REQUIRES_X86_F16C;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__f16c_polynomial_p19h9t2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_expm1minus_rr1_p3h2ts_div, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_expm1minus_rr1_p3h2ts_div, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__fma3_expm1minus_rr1_p3h2ts_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_expm1minus_rr1_p3h2ts_rcp, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_expm1minus_rr1_p3h2ts_rcp, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__fma3_expm1minus_rr1_p3h2ts_rcp);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_POLYNOMIAL_P17H8T2, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_polynomial_p17h8t2, 1.0f);
  }

  TEST(TANH__FMA3_POLYNOMIAL_P17H8T2, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_polynomial_p17h8t2, -1.0f);
  }

  TEST(TANH__FMA3_POLYNOMIAL_P17H8T2, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__fma3_polynomial_p17h8t2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_POLYNOMIAL_P19H9T2, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_polynomial_p19h9t2, 1.0f);
  }

  TEST(TANH__FMA3_POLYNOMIAL_P19H9T2, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__fma3_polynomial_p19h9t2, -1.0f);
  }

  TEST(TANH__FMA3_POLYNOMIAL_P19H9T2, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__fma3_polynomial_p19h9t2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__avx2_expm1minus_rr1_p3h2ts_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__avx2_expm1minus_rr1_p3h2ts_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__avx2_expm1minus_rr1_p3h2ts_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.208p+2f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f16_tanh__avx2_expm1minus_rr1_p3h2ts_rcp, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.208p+2f)
      .TestOutputMatchReference(xnn_math_f16_tanh__avx2_expm1minus_rr1_p3h2ts_rcp, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f16_tanh__avx2_expm1minus_rr1_p3h2ts_rcp);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
