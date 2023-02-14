// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits>

#include <gtest/gtest.h>

#include "math-evaluation-tester.h"

#include <xnnpack/isa-checks.h>


#if XNN_ARCH_ARM64
  TEST(TANH__AARCH64_NEONFMA_EXPM1_RR1_P6H5_DIV, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__aarch64_neonfma_expm1_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__AARCH64_NEONFMA_EXPM1_RR1_P6H5_DIV, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__aarch64_neonfma_expm1_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__AARCH64_NEONFMA_EXPM1_RR1_P6H5_DIV, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__aarch64_neonfma_expm1_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1_RR1_P6H5_NR1RECPS1FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1_rr1_p6h5_nr1recps1fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1_RR1_P6H5_NR1RECPS1FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1_rr1_p6h5_nr1recps1fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1_RR1_P6H5_NR1RECPS1FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1_rr1_p6h5_nr1recps1fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1_RR1_P6H5_NR2FMA, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1_rr1_p6h5_nr2fma, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1_RR1_P6H5_NR2FMA, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1_rr1_p6h5_nr2fma, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1_RR1_P6H5_NR2FMA, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1_rr1_p6h5_nr2fma);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEONFMA_EXPM1_RR1_P6H5_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1_rr1_p6h5_nr2recps, 1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1_RR1_P6H5_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neonfma_expm1_rr1_p6h5_nr2recps, -1.0f);
  }

  TEST(TANH__NEONFMA_EXPM1_RR1_P6H5_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neonfma_expm1_rr1_p6h5_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEON_EXPM1_RR1_P6H5_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1_rr1_p6h5_nr2recps, 1.0f);
  }

  TEST(TANH__NEON_EXPM1_RR1_P6H5_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1_rr1_p6h5_nr2recps, -1.0f);
  }

  TEST(TANH__NEON_EXPM1_RR1_P6H5_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neon_expm1_rr1_p6h5_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(TANH__NEON_EXPM1_RR2_P6H5_NR2RECPS, positive_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1_rr2_p6h5_nr2recps, 1.0f);
  }

  TEST(TANH__NEON_EXPM1_RR2_P6H5_NR2RECPS, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__neon_expm1_rr2_p6h5_nr2recps, -1.0f);
  }

  TEST(TANH__NEON_EXPM1_RR2_P6H5_NR2RECPS, nan) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__neon_expm1_rr2_p6h5_nr2recps);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512F_EXPM1_RR1_LUT4_P4H3_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4h3_perm_div, 1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_LUT4_P4H3_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4h3_perm_div, -1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_LUT4_P4H3_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4h3_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512F_EXPM1_RR1_LUT4_P4H3_PERM_NR1FMA, positive_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4h3_perm_nr1fma, 1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_LUT4_P4H3_PERM_NR1FMA, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4h3_perm_nr1fma, -1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_LUT4_P4H3_PERM_NR1FMA, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4h3_perm_nr1fma);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512F_EXPM1_RR1_LUT4_P4H3_PERM_NR1FMA1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4h3_perm_nr1fma1adj, 1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_LUT4_P4H3_PERM_NR1FMA1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4h3_perm_nr1fma1adj, -1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_LUT4_P4H3_PERM_NR1FMA1ADJ, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4h3_perm_nr1fma1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512F_EXPM1_RR1_P6H5_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_P6H5_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_P6H5_DIV, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512f_expm1_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512F_EXPM1_RR1_P6H5_NR1FMA, positive_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_p6h5_nr1fma, 1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_P6H5_NR1FMA, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_p6h5_nr1fma, -1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_P6H5_NR1FMA, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512f_expm1_rr1_p6h5_nr1fma);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX512F_EXPM1_RR1_P6H5_NR1FMA1ADJ, positive_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_p6h5_nr1fma1adj, 1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_P6H5_NR1FMA1ADJ, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx512f_expm1_rr1_p6h5_nr1fma1adj, -1.0f);
  }

  TEST(TANH__AVX512F_EXPM1_RR1_P6H5_NR1FMA1ADJ, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx512f_expm1_rr1_p6h5_nr1fma1adj);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1_RR1_LUT4_P4H3_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1_rr1_lut4_p4h3_perm_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1_RR1_LUT4_P4H3_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1_rr1_lut4_p4h3_perm_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1_RR1_LUT4_P4H3_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1_rr1_lut4_p4h3_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1_RR1_LUT8_P4H3_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1_rr1_lut8_p4h3_perm_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1_RR1_LUT8_P4H3_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1_rr1_lut8_p4h3_perm_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1_RR1_LUT8_P4H3_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1_rr1_lut8_p4h3_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX2_EXPM1_RR1_P6H5_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__AVX2_EXPM1_RR1_P6H5_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx2_expm1_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__AVX2_EXPM1_RR1_P6H5_DIV, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx2_expm1_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1_RR1_LUT4_P4H3_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1_rr1_lut4_p4h3_perm_div, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1_RR1_LUT4_P4H3_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1_rr1_lut4_p4h3_perm_div, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1_RR1_LUT4_P4H3_PERM_DIV, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1_rr1_lut4_p4h3_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__FMA3_EXPM1_RR1_P6H5_DIV, positive_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__FMA3_EXPM1_RR1_P6H5_DIV, negative_saturation) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__fma3_expm1_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__FMA3_EXPM1_RR1_P6H5_DIV, nan) {
    TEST_REQUIRES_X86_FMA3;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__fma3_expm1_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1_RR1_LUT4_P4H2_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1_rr1_lut4_p4h2_perm_div, 1.0f);
  }

  TEST(TANH__AVX_EXPM1_RR1_LUT4_P4H2_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1_rr1_lut4_p4h2_perm_div, -1.0f);
  }

  TEST(TANH__AVX_EXPM1_RR1_LUT4_P4H2_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1_rr1_lut4_p4h2_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1_RR1_LUT4_P4H3_PERM_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1_rr1_lut4_p4h3_perm_div, 1.0f);
  }

  TEST(TANH__AVX_EXPM1_RR1_LUT4_P4H3_PERM_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1_rr1_lut4_p4h3_perm_div, -1.0f);
  }

  TEST(TANH__AVX_EXPM1_RR1_LUT4_P4H3_PERM_DIV, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1_rr1_lut4_p4h3_perm_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__AVX_EXPM1_RR1_P6H5_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__AVX_EXPM1_RR1_P6H5_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__avx_expm1_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__AVX_EXPM1_RR1_P6H5_DIV, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__avx_expm1_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(TANH__SSE2_EXPM1_RR1_P6H5_DIV, positive_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1_rr1_p6h5_div, 1.0f);
  }

  TEST(TANH__SSE2_EXPM1_RR1_P6H5_DIV, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__sse2_expm1_rr1_p6h5_div, -1.0f);
  }

  TEST(TANH__SSE2_EXPM1_RR1_P6H5_DIV, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__sse2_expm1_rr1_p6h5_div);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_ABS_MIN, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_abs_min, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_ABS_MIN, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_abs_min, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_ABS_MIN, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_abs_min);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_ABS_PMIN, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_abs_pmin, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_ABS_PMIN, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_abs_pmin, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_ABS_PMIN, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_abs_pmin);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_NABS_MAX, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_nabs_max, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_NABS_MAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_nabs_max, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_NABS_MAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_nabs_max);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_NABS_PMAX, positive_saturation) {
    MathEvaluationTester()
      .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_nabs_pmax, 1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_NABS_PMAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
      .TestOutputMatchReference(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_nabs_pmax, -1.0f);
  }

  TEST(TANH__WASMSIMD_EXPM1_RR1_P6H5_DIV_NABS_PMAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6h5_div_nabs_pmax);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(TANH__FMA_EXPM1_RR1_LUT4_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut4_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT4_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut4_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT4_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1_rr1_lut4_p4h3_div);
}


TEST(TANH__FMA_EXPM1_RR1_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut8_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut8_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1_rr1_lut8_p3h1_div);
}


TEST(TANH__FMA_EXPM1_RR1_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut8_p4h3_div, 1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut8_p4h3_div, -1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1_rr1_lut8_p4h3_div);
}


TEST(TANH__FMA_EXPM1_RR1_LUT16_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut16_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT16_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut16_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT16_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1_rr1_lut16_p3h1_div);
}


TEST(TANH__FMA_EXPM1_RR1_LUT16_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut16_p4h2_div, 1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT16_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut16_p4h2_div, -1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT16_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1_rr1_lut16_p4h2_div);
}


TEST(TANH__FMA_EXPM1_RR1_LUT32_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut32_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT32_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut32_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT32_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1_rr1_lut32_p3h1_div);
}


TEST(TANH__FMA_EXPM1_RR1_LUT64_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut64_p3h1_div, 1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT64_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_lut64_p3h1_div, -1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_LUT64_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1_rr1_lut64_p3h1_div);
}


TEST(TANH__FMA_EXPM1_RR1_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_p6h5_div, 1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__fma_expm1_rr1_p6h5_div, -1.0f);
}

TEST(TANH__FMA_EXPM1_RR1_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__fma_expm1_rr1_p6h5_div);
}


TEST(TANH__SCALAR_EXPM1_RR1_LUT4_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut4_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT4_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut4_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT4_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr1_lut4_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1_RR1_LUT4_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut4_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT4_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut4_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT4_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr1_lut4_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1_RR1_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut8_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut8_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr1_lut8_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1_RR2_LUT8_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr2_lut8_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR2_LUT8_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr2_lut8_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR2_LUT8_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr2_lut8_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1_RR1_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut8_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut8_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr1_lut8_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1_RR2_LUT8_P4H3_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr2_lut8_p4h3_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR2_LUT8_P4H3_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr2_lut8_p4h3_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR2_LUT8_P4H3_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr2_lut8_p4h3_div);
}


TEST(TANH__SCALAR_EXPM1_RR1_LUT16_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut16_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT16_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut16_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT16_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr1_lut16_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1_RR1_LUT16_P4H2_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut16_p4h2_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT16_P4H2_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut16_p4h2_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT16_P4H2_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr1_lut16_p4h2_div);
}


TEST(TANH__SCALAR_EXPM1_RR1_LUT32_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut32_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT32_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut32_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT32_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr1_lut32_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1_RR1_LUT64_P3H1_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut64_p3h1_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT64_P3H1_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_lut64_p3h1_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_LUT64_P3H1_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr1_lut64_p3h1_div);
}


TEST(TANH__SCALAR_EXPM1_RR1_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_p6h5_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr1_p6h5_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR1_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr1_p6h5_div);
}


TEST(TANH__SCALAR_EXPM1_RR2_P6H5_DIV, positive_saturation) {
  MathEvaluationTester()
    .input_range(0x1.205968p+3f, std::numeric_limits<float>::infinity())
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr2_p6h5_div, 1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR2_P6H5_DIV, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.205968p+3f)
    .TestOutputMatchReference(xnn_math_f32_tanh__scalar_expm1_rr2_p6h5_div, -1.0f);
}

TEST(TANH__SCALAR_EXPM1_RR2_P6H5_DIV, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_tanh__scalar_expm1_rr2_p6h5_div);
}
