// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits>

#include <gtest/gtest.h>

#include "math-evaluation-tester.h"

#include <xnnpack/isa-checks.h>


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPMINUS__NEONFMA_RR2_LUT64_P2, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_expminus__neonfma_rr2_lut64_p2, 1.0f);
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT64_P2, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_expminus__neonfma_rr2_lut64_p2, 1.0f);
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT64_P2, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.5D58A0p+6f)
      .TestOutputMatchReference(xnn_math_f32_expminus__neonfma_rr2_lut64_p2, 0.0f);
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT64_P2, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expminus__neonfma_rr2_lut64_p2);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPMINUS__NEONFMA_RR2_LUT2048_P1, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_expminus__neonfma_rr2_lut2048_p1, 1.0f);
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT2048_P1, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_expminus__neonfma_rr2_lut2048_p1, 1.0f);
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT2048_P1, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.5D58A0p+6f)
      .TestOutputMatchReference(xnn_math_f32_expminus__neonfma_rr2_lut2048_p1, 0.0f);
  }

  TEST(EXPMINUS__NEONFMA_RR2_LUT2048_P1, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expminus__neonfma_rr2_lut2048_p1);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPMINUS__NEONFMA_RR2_P5, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_expminus__neonfma_rr2_p5, 1.0f);
  }

  TEST(EXPMINUS__NEONFMA_RR2_P5, positive_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_expminus__neonfma_rr2_p5, 1.0f);
  }

  TEST(EXPMINUS__NEONFMA_RR2_P5, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.5D58A0p+6f)
      .TestOutputMatchReference(xnn_math_f32_expminus__neonfma_rr2_p5, 0.0f);
  }

  TEST(EXPMINUS__NEONFMA_RR2_P5, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expminus__neonfma_rr2_p5);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPMINUS__AVX2_RR2_P5, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_expminus__avx2_rr2_p5, 1.0f);
  }

  TEST(EXPMINUS__AVX2_RR2_P5, positive_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_expminus__avx2_rr2_p5, 1.0f);
  }

  TEST(EXPMINUS__AVX2_RR2_P5, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.5D58A0p+6f)
      .TestOutputMatchReference(xnn_math_f32_expminus__avx2_rr2_p5, 0.0f);
  }

  TEST(EXPMINUS__AVX2_RR2_P5, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expminus__avx2_rr2_p5);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPMINUS__SSE2_RR2_P5, negative_zero) {
    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchReference(xnn_math_f32_expminus__sse2_rr2_p5, 1.0f);
  }

  TEST(EXPMINUS__SSE2_RR2_P5, positive_zero) {
    MathEvaluationTester()
      .input_value(+0.0f)
      .TestOutputMatchReference(xnn_math_f32_expminus__sse2_rr2_p5, 1.0f);
  }

  TEST(EXPMINUS__SSE2_RR2_P5, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.5D58A0p+6f)
      .TestOutputMatchReference(xnn_math_f32_expminus__sse2_rr2_p5, 0.0f);
  }

  TEST(EXPMINUS__SSE2_RR2_P5, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expminus__sse2_rr2_p5);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(EXPMINUS__SCALAR_RR2_LUT64_P2, negative_zero) {
  MathEvaluationTester()
    .input_value(-0.0f)
    .TestOutputMatchReference(xnn_math_f32_expminus__scalar_rr2_lut64_p2, 1.0f);
}

TEST(EXPMINUS__SCALAR_RR2_LUT64_P2, positive_zero) {
  MathEvaluationTester()
    .input_value(+0.0f)
    .TestOutputMatchReference(xnn_math_f32_expminus__scalar_rr2_lut64_p2, 1.0f);
}

TEST(EXPMINUS__SCALAR_RR2_LUT64_P2, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.5D58A0p+6f)
    .TestOutputMatchReference(xnn_math_f32_expminus__scalar_rr2_lut64_p2, 0.0f);
}

TEST(EXPMINUS__SCALAR_RR2_LUT64_P2, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_expminus__scalar_rr2_lut64_p2);
}


TEST(EXPMINUS__SCALAR_RR2_LUT2048_P1, negative_zero) {
  MathEvaluationTester()
    .input_value(-0.0f)
    .TestOutputMatchReference(xnn_math_f32_expminus__scalar_rr2_lut2048_p1, 1.0f);
}

TEST(EXPMINUS__SCALAR_RR2_LUT2048_P1, positive_zero) {
  MathEvaluationTester()
    .input_value(+0.0f)
    .TestOutputMatchReference(xnn_math_f32_expminus__scalar_rr2_lut2048_p1, 1.0f);
}

TEST(EXPMINUS__SCALAR_RR2_LUT2048_P1, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.5D58A0p+6f)
    .TestOutputMatchReference(xnn_math_f32_expminus__scalar_rr2_lut2048_p1, 0.0f);
}

TEST(EXPMINUS__SCALAR_RR2_LUT2048_P1, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_expminus__scalar_rr2_lut2048_p1);
}


TEST(EXPMINUS__SCALAR_RR2_P5, negative_zero) {
  MathEvaluationTester()
    .input_value(-0.0f)
    .TestOutputMatchReference(xnn_math_f32_expminus__scalar_rr2_p5, 1.0f);
}

TEST(EXPMINUS__SCALAR_RR2_P5, positive_zero) {
  MathEvaluationTester()
    .input_value(+0.0f)
    .TestOutputMatchReference(xnn_math_f32_expminus__scalar_rr2_p5, 1.0f);
}

TEST(EXPMINUS__SCALAR_RR2_P5, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.5D58A0p+6f)
    .TestOutputMatchReference(xnn_math_f32_expminus__scalar_rr2_p5, 0.0f);
}

TEST(EXPMINUS__SCALAR_RR2_P5, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_expminus__scalar_rr2_p5);
}
