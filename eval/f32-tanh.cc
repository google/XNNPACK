// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: eval/f32-tanh.yaml
//   Generator: tools/generate-tanh-eval.py


#include <limits>

#include <gtest/gtest.h>

#include "math-evaluation-tester.h"

#include <xnnpack/isa-checks.h>


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2TS_NR1RECPS1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr1recps1fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2TS_NR1RECPS1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr1recps1fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2TS_NR1RECPS1FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr1recps1fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2TS_NR2FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr2fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2TS_NR2FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr2fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2TS_NR2FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr2fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2TS_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr2recps, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2TS_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr2recps, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H2TS_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H2TS_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h2ts_nr2recps, 1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H2TS_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h2ts_nr2recps, -1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H2TS_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h2ts_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ps_div, 1.0f);
  }

  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ps_div, -1.0f);
  }

  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ps_div);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1RECPS1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1RECPS1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1RECPS1FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1RECPS1FMAADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fmaadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1RECPS1FMAADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fmaadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1RECPS1FMAADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fmaadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2FMAADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2fmaadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2FMAADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2fmaadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2FMAADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2fmaadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2recps, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2recps, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2RECPSADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2recpsadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2RECPSADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2recpsadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3PS_NR2RECPSADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2recpsadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H3PS_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h3ps_nr2recps, 1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H3PS_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h3ps_nr2recps, -1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR2_LUT8_P4H3PS_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h3ps_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_p6h5ts_div, 1.0f);
  }

  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_p6h5ts_div, -1.0f);
  }

  TEST(TANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_p6h5ts_div);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMAADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fmaadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMAADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fmaadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMAADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fmaadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMAADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2fmaadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMAADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2fmaadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMAADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2fmaadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2recps, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2recps, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPSADJ, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2recpsadj, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPSADJ, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2recpsadj, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPSADJ, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2recpsadj);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr1_p6h5ts_nr2recps, 1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1minus_rr1_p6h5ts_nr2recps, -1.0f);
  }

  TEST(TANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neon_expm1minus_rr1_p6h5ts_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2TS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2ts_nr1, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2TS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2ts_nr1, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2TS_NR1, nan) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2ts_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2TS_NR2, positive_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2ts_nr2, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2TS_NR2, negative_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2ts_nr2, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H2TS_NR2, nan) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2ts_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_lut8_p4h3ps_div, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_lut8_p4h3ps_div, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, nan) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr1_lut8_p4h3ps_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3PS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ps_nr1, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3PS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ps_nr1, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3PS_NR1, nan) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ps_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3PS_NR2, positive_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ps_nr2, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3PS_NR2, negative_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ps_nr2, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3PS_NR2, nan) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ps_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3TS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ts_nr1, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3TS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ts_nr1, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3TS_NR1, nan) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ts_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3TS_NR2, positive_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ts_nr2, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3TS_NR2, negative_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ts_nr2, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR2_LUT8_P4H3TS_NR2, nan) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ts_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_div, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_div, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV, nan) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_nr1, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_nr1, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1, nan) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2, positive_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_nr2, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2, negative_saturation) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_nr2, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2, nan) {
    TEST_REQUIRES_X86_SSE2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_lut4_p4h2ts_perm_div, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_lut4_p4h2ts_perm_div, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr1_lut4_p4h2ts_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H2TS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h2ts_nr1, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H2TS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h2ts_nr1, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H2TS_NR1, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h2ts_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H2TS_NR2, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h2ts_nr2, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H2TS_NR2, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h2ts_nr2, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H2TS_NR2, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h2ts_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_lut8_p4h3ps_div, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_lut8_p4h3ps_div, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr1_lut8_p4h3ps_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3PS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ps_nr1, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3PS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ps_nr1, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3PS_NR1, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ps_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3PS_NR2, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ps_nr2, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3PS_NR2, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ps_nr2, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3PS_NR2, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ps_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3TS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ts_nr1, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3TS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ts_nr1, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3TS_NR1, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ts_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3TS_NR2, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ts_nr2, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3TS_NR2, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ts_nr2, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR2_LUT8_P4H3TS_NR2, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ts_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_div, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_div, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_nr1, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_nr1, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2, positive_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_nr2, 1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2, negative_saturation) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_nr2, -1.0f);
  }

  TEST(TANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2, nan) {
    TEST_REQUIRES_X86_AVX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_nr2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_div, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_div, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_nr1, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_nr1, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_nr1adj, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_nr1adj, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3PS_NR1ADJ, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_div, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_div, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_nr1, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_nr1, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_nr1adj, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_nr1adj, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ, nan) {
    TEST_REQUIRES_X86_FMA3;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_nr1, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_nr1, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_nr1adj, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_nr1adj, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_nr1, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_nr1, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_nr1adj, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_nr1adj, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_nr1, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_nr1, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_nr1adj, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_nr1adj, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX2;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_div, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_div, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_nr1, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_nr1, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_nr1adj, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_nr1adj, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_PERM_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_div, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_div, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_DIV, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_nr1, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_nr1, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_nr1adj, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_nr1adj, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3PS_GATHER_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_div, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_div, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_nr1, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_nr1, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_nr1);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_nr1adj, 1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_nr1adj, -1.0f);
  }

  TEST(TANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1ADJ, nan) {
    TEST_REQUIRES_X86_AVX512SKX;
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_nr1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_ABS_MIN, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_abs_min, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_ABS_MIN, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_abs_min, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_ABS_MIN, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_abs_min);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_ABS_PMIN, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_abs_pmin, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_ABS_PMIN, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_abs_pmin, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_ABS_PMIN, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_abs_pmin);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_NABS_MAX, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_max, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_NABS_MAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_max, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_NABS_MAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_max);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_NABS_PMAX, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_pmax, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_NABS_PMAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_pmax, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV_NABS_PMAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_pmax);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p4h3ps_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p4h3ps_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_lut8_p4h3ps_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_p6h5ts_div, 1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasm_expm1minus_rr1_p6h5ts_div, -1.0f);
  }

  TEST(TANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasm_expm1minus_rr1_p6h5ts_div);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2ts_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2ts_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H2TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2ts_rcp);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT4_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT4_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2ts_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2ts_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H2TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2ts_rcp);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2ts_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2ts_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H2TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2ts_rcp);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3PS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ps_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3PS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ps_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3PS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ps_rcp);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ts_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ts_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ts_rcp);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3PS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ps_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3PS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ps_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3PS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ps_rcp);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ts_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ts_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT8_P4H3TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ts_rcp);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2ts_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2ts_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H2TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2ts_rcp);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT16_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT16_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT32_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut32_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT32_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut32_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT32_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut32_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT32_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut32_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT32_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut32_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT32_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut32_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT64_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut64_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT64_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut64_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_LUT64_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_lut64_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT64_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut64_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT64_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut64_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_LUT64_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_lut64_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H4TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h4ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H4TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h4ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H4TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h4ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H4TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h4ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H4TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h4ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H4TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h4ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ps_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ts_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5PS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ps_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5PS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ps_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5PS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ps_rcp);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ts_rcp, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ts_rcp, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ts_rcp);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H5PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H5PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H5PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5ps_div);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H5TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H5TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1MINUS_RR2_P6H5TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT4_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT4_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT8_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT8_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h2ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h2ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h2ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT16_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3ps_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT16_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT32_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut32_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT32_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut32_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT32_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut32_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT32_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut32_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT32_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut32_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT32_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut32_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT64_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut64_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT64_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut64_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_LUT64_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_lut64_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT64_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut64_p3h1ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT64_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut64_p3h1ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_LUT64_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_lut64_p3h1ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H4TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h4ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H4TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h4ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H4TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h4ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H4TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h4ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H4TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h4ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H4TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h4ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H5PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H5PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H5PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5ps_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H5TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H5TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR1_P6H5TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5ts_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H5PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5ps_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H5PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5ps_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H5PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5ps_div);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H5TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5ts_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H5TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5ts_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1PLUS_RR2_P6H5TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2ts_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2ts_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H2TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2ts_rcp);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3PS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ps_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3PS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ps_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3PS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ps_rcp);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ts_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ts_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT4_P4H3TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ts_rcp);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT4_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2ts_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2ts_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H2TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2ts_rcp);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2ts_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2ts_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H2TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2ts_rcp);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3PS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ps_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3PS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ps_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3PS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ps_rcp);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ts_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ts_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ts_rcp);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT8_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2ts_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2ts_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H2TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2ts_rcp);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT16_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT16_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT32_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut32_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT32_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut32_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT32_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut32_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT32_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut32_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT32_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut32_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT32_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut32_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT64_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut64_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT64_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_lut64_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_LUT64_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_lut64_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT64_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut64_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT64_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_lut64_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_LUT64_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_lut64_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H4TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h4ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H4TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h4ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H4TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h4ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H4TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h4ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H4TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h4ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H4TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h4ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ps_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ts_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5PS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ps_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5PS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ps_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5PS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ps_rcp);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5TS_RCP, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ts_rcp, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5TS_RCP, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ts_rcp, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR1_P6H5TS_RCP, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ts_rcp);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H5PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H5PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H5PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5ps_div);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H5TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H5TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1MINUS_RR2_P6H5TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT4_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT4_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT8_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT8_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H2TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h2ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H2TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h2ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H2TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h2ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT16_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H3PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H3PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H3PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3ps_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H3TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H3TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT16_P4H3TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT32_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut32_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT32_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut32_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT32_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut32_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT32_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut32_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT32_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut32_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT32_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut32_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT64_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut64_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT64_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_lut64_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_LUT64_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_lut64_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT64_P3H1TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut64_p3h1ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT64_P3H1TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_lut64_p3h1ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_LUT64_P3H1TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_lut64_p3h1ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H4TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h4ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H4TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h4ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H4TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h4ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H4TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h4ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H4TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h4ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H4TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h4ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H5PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H5PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H5PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5ps_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H5TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H5TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR1_P6H5TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5ts_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H5PS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5ps_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H5PS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5ps_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H5PS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5ps_div);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H5TS_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5ts_div, 1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H5TS_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5ts_div, -1.0f);
}

TEST(TANH__FMA_EXPM1PLUS_RR2_P6H5TS_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5ts_div);
}