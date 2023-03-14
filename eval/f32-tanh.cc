// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits>

#include <gtest/gtest.h>

#include "math-evaluation-tester.h"

#include <xnnpack/isa-checks.h>


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2_NR1RECPS1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2_nr1recps1fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2_NR1RECPS1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2_nr1recps1fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2_NR1RECPS1FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2_nr1recps1fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2_NR2FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2_nr2fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2_NR2FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2_nr2fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2_NR2FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2_nr2fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2_nr2recps, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2_nr2recps, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H2_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h2_nr2recps, 1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H2_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h2_nr2recps, -1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H2_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h2_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_lut8_p4h3_div, 1.0f);
  }

  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_lut8_p4h3_div, -1.0f);
  }

  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_lut8_p4h3_div);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR1RECPS1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr1recps1fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR1RECPS1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr1recps1fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR1RECPS1FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr1recps1fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR1RECPS1FMAADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr1recps1fmaadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR1RECPS1FMAADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr1recps1fmaadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR1RECPS1FMAADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr1recps1fmaadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2FMAADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2fmaadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2FMAADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2fmaadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2FMAADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2fmaadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2recps, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2recps, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2RECPSADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2recpsadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2RECPSADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2recpsadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3_NR2RECPSADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3_nr2recpsadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H3_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h3_nr2recps, 1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H3_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h3_nr2recps, -1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H3_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h3_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR1RECPS1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr1recps1fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR1RECPS1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr1recps1fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR1RECPS1FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr1recps1fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR1RECPS1FMAADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr1recps1fmaadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR1RECPS1FMAADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr1recps1fmaadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR1RECPS1FMAADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr1recps1fmaadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2FMAADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2fmaadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2FMAADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2fmaadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2FMAADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2fmaadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2recps, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2recps, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2RECPSADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2recpsadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2RECPSADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2recpsadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5_NR2RECPSADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5_nr2recpsadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEON_EXPM1MINUS_RR1_P6H5_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr1_p6h5_nr2recps, 1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR1_P6H5_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr1_p6h5_nr2recps, -1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR1_P6H5_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neon_expm1minus_rr1_p6h5_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H2_GATHER_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h2_gather_nr1, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H2_GATHER_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h2_gather_nr1, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H2_GATHER_NR1, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h2_gather_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H2_PERM_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h2_perm_nr1, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H2_PERM_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h2_perm_nr1, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H2_PERM_NR1, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h2_perm_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5_nr1, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5_nr1, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5_nr1adj, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5_nr1adj, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX512SKX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3_perm_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3_perm_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H2_GATHER_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h2_gather_nr1, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H2_GATHER_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h2_gather_nr1, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H2_GATHER_NR1, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h2_gather_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H2_PERM_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h2_perm_nr1, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H2_PERM_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h2_perm_nr1, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H2_PERM_NR1, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h2_perm_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_gather_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_gather_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_gather_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_perm_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_perm_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5_nr1, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5_nr1, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5_nr1adj, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5_nr1adj, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3_perm_div, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3_perm_div, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H2_NR1, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h2_nr1, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H2_NR1, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h2_nr1, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H2_NR1, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h2_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3_div, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3_div, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3_nr1, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3_nr1, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3_nr1adj, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3_nr1adj, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5_nr1, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5_nr1, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5_nr1adj, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5_nr1adj, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_lut4_p4h2_perm_div, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_lut4_p4h2_perm_div, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr1_lut4_p4h2_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_lut8_p4h3_div, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_lut8_p4h3_div, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr1_lut8_p4h3_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5_DIV, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5_nr1, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5_nr1, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5_NR1, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5_NR2, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5_nr2, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5_NR2, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5_nr2, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5_NR2, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2_nr1, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2_nr1, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2_NR1, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2_NR2, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2_nr2, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2_NR2, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2_nr2, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2_NR2, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_lut8_p4h3_div, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_lut8_p4h3_div, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr1_lut8_p4h3_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3_nr1, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3_nr1, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3_NR1, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3_NR2, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3_nr2, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3_NR2, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3_nr2, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3_NR2, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5_nr1, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5_nr1, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5_nr2, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5_nr2, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_ABS_MIN, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_abs_min, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_ABS_MIN, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_abs_min, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_ABS_MIN, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_abs_min);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_ABS_PMIN, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_abs_pmin, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_ABS_PMIN, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_abs_pmin, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_ABS_PMIN, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_abs_pmin);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_NABS_MAX, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_nabs_max, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_NABS_MAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_nabs_max, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_NABS_MAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_nabs_max);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_NABS_PMAX, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_nabs_pmax, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_NABS_PMAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_nabs_pmax, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3_DIV_NABS_PMAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3_div_nabs_pmax);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_ABS_MIN, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_abs_min, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_ABS_MIN, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_abs_min, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_ABS_MIN, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_abs_min);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_ABS_PMIN, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_abs_pmin, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_ABS_PMIN, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_abs_pmin, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_ABS_PMIN, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_abs_pmin);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_NABS_MAX, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_nabs_max, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_NABS_MAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_nabs_max, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_NABS_MAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_nabs_max);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_NABS_PMAX, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_nabs_pmax, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_NABS_PMAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_nabs_pmax, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5_DIV_NABS_PMAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5_div_nabs_pmax);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT4_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut4_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT4_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut4_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT4_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut4_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT4_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut4_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT4_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut4_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT4_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut4_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT4_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut4_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT4_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut4_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT4_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut4_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT4_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut4_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT4_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut4_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT4_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut4_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT8_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut8_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT8_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut8_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT8_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut8_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT8_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut8_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT8_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut8_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT8_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut8_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT8_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut8_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT8_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut8_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT8_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut8_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT16_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut16_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT16_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut16_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT16_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut16_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT16_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut16_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT16_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut16_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT16_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut16_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT16_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut16_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT16_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut16_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT16_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut16_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT16_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut16_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT16_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut16_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT16_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut16_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT16_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut16_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT16_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut16_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT16_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut16_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT16_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut16_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT16_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut16_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT16_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut16_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT32_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut32_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT32_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut32_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT32_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut32_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT32_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut32_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT32_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut32_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT32_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut32_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT64_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut64_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT64_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut64_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT64_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut64_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT64_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut64_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT64_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut64_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_LUT64_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_lut64_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_P6H4_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_p6h4_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_P6H4_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_p6h4_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_P6H4_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_p6h4_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_P6H4_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_p6h4_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_P6H4_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_p6h4_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_P6H4_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_p6h4_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_P6H5_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_P6H5_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_P6H5_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR2_P6H5_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_p6h5_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_P6H5_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr2_p6h5_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR2_P6H5_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr2_p6h5_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT4_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut4_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT4_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut4_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT4_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut4_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT4_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut4_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT4_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut4_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT4_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut4_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT4_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut4_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT4_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut4_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT4_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut4_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT4_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut4_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT4_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut4_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT4_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut4_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT8_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut8_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT8_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut8_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT8_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut8_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT8_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut8_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT8_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut8_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT8_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut8_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT8_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut8_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT8_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut8_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT8_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut8_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT8_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut8_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT8_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut8_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT8_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut8_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT8_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut8_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT8_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut8_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT8_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut8_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT8_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut8_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT8_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut8_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT8_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut8_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT16_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut16_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT16_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut16_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT16_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut16_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT16_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut16_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT16_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut16_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT16_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut16_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT16_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut16_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT16_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut16_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT16_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut16_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT16_P4H2_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut16_p4h2_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT16_P4H2_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut16_p4h2_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT16_P4H2_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut16_p4h2_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT16_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut16_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT16_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut16_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT16_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut16_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT16_P4H3_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut16_p4h3_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT16_P4H3_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut16_p4h3_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT16_P4H3_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut16_p4h3_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT32_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut32_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT32_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut32_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT32_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut32_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT32_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut32_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT32_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut32_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT32_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut32_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT64_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut64_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT64_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut64_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_LUT64_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_lut64_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT64_P3H1_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut64_p3h1_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT64_P3H1_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut64_p3h1_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_LUT64_P3H1_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_lut64_p3h1_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_P6H4_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_p6h4_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_P6H4_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_p6h4_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_P6H4_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_p6h4_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_P6H4_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_p6h4_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_P6H4_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_p6h4_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_P6H4_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_p6h4_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR1_P6H5_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_P6H5_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR1_P6H5_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1PLUS_RR2_P6H5_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_p6h5_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_P6H5_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1plus_rr2_p6h5_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1PLUS_RR2_P6H5_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1plus_rr2_p6h5_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2_rcp);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h2_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3_rcp);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p3h1_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p3h1_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2_rcp);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2_rcp);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3_rcp);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p3h1_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p3h1_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2_rcp);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h2_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT32_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut32_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT32_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut32_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT32_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut32_p3h1_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT32_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut32_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT32_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut32_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT32_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut32_p3h1_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_LUT64_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut64_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT64_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut64_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT64_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut64_p3h1_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_LUT64_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut64_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT64_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut64_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT64_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut64_p3h1_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_P6H4_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h4_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H4_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h4_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H4_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h4_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_P6H4_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h4_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H4_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h4_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H4_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h4_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5_div);
}


TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5_rcp);
}


TEST(TANH__FMA_EXPM1MINUS_RR2_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h2_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h2_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p3h1_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p3h1_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h2_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h2_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p3h1_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p3h1_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h2_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h2_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_LUT32_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut32_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT32_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut32_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT32_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut32_p3h1_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_LUT32_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut32_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT32_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut32_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT32_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut32_p3h1_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_LUT64_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut64_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT64_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut64_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT64_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut64_p3h1_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_LUT64_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut64_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT64_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut64_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT64_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut64_p3h1_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_P6H4_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h4_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H4_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h4_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H4_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h4_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_P6H4_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h4_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H4_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h4_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H4_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h4_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR1_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5_div);
}


TEST(TANH__FMA_EXPM1PLUS_RR2_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2_rcp);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2_rcp);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2_rcp);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3_rcp);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3_rcp);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2_rcp);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT32_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut32_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT32_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut32_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT32_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut32_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT32_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut32_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT32_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut32_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT32_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut32_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT64_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut64_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT64_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut64_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT64_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut64_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT64_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut64_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT64_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut64_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT64_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut64_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H4_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h4_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H4_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h4_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H4_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h4_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H4_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h4_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H4_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h4_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H4_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h4_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5_div);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5_rcp);
}


TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT32_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut32_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT32_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut32_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT32_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut32_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT32_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut32_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT32_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut32_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT32_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut32_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT64_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut64_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT64_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut64_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT64_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut64_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT64_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut64_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT64_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut64_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT64_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut64_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H4_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h4_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H4_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h4_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H4_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h4_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H4_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h4_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H4_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h4_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H4_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h4_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5_div);
}


TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5_div);
}
