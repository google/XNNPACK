// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits>

#include <gtest/gtest.h>

#include "math-evaluation-tester.h"

#include <xnnpack/isa-checks.h>


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPM1MINUS__NEON_RR2_LUT16_P3, negative_zero) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__neon_rr2_lut16_p3);
  }

  TEST(EXPM1MINUS__NEON_RR2_LUT16_P3, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__neon_rr2_lut16_p3, -1.0f);
  }

  TEST(EXPM1MINUS__NEON_RR2_LUT16_P3, nan) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__neon_rr2_lut16_p3);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPM1MINUS__NEON_RR2_P6, negative_zero) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__neon_rr2_p6);
  }

  TEST(EXPM1MINUS__NEON_RR2_P6, negative_saturation) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__neon_rr2_p6, -1.0f);
  }

  TEST(EXPM1MINUS__NEON_RR2_P6, nan) {
    TEST_REQUIRES_ARM_NEON;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__neon_rr2_p6);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPM1MINUS__NEONFMA_RR1_LUT16_P3, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3);
  }

  TEST(EXPM1MINUS__NEONFMA_RR1_LUT16_P3, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3, -1.0f);
  }

  TEST(EXPM1MINUS__NEONFMA_RR1_LUT16_P3, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(EXPM1MINUS__NEONFMA_RR1_P6, negative_zero) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__neonfma_rr1_p6);
  }

  TEST(EXPM1MINUS__NEONFMA_RR1_P6, negative_saturation) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__neonfma_rr1_p6, -1.0f);
  }

  TEST(EXPM1MINUS__NEONFMA_RR1_P6, nan) {
    TEST_REQUIRES_ARM_NEON_FMA;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__neonfma_rr1_p6);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX512F_RR1_LUT16_P3_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm);
  }

  TEST(EXPM1MINUS__AVX512F_RR1_LUT16_P3_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm, -1.0f);
  }

  TEST(EXPM1MINUS__AVX512F_RR1_LUT16_P3_PERM, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX512F_RR1_P6, negative_zero) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__avx512f_rr1_p6);
  }

  TEST(EXPM1MINUS__AVX512F_RR1_P6, negative_saturation) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__avx512f_rr1_p6, -1.0f);
  }

  TEST(EXPM1MINUS__AVX512F_RR1_P6, nan) {
    TEST_REQUIRES_X86_AVX512F;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__avx512f_rr1_p6);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX2_RR1_LUT4_P4_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm);
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT4_P4_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm, -1.0f);
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT4_P4_PERM, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX2_RR1_LUT8_P4_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__avx2_rr1_lut8_p4_perm);
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT8_P4_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__avx2_rr1_lut8_p4_perm, -1.0f);
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT8_P4_PERM, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__avx2_rr1_lut8_p4_perm);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX2_RR1_LUT16_P3_GATHER, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__avx2_rr1_lut16_p3_gather);
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT16_P3_GATHER, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__avx2_rr1_lut16_p3_gather, -1.0f);
  }

  TEST(EXPM1MINUS__AVX2_RR1_LUT16_P3_GATHER, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__avx2_rr1_lut16_p3_gather);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX2_RR1_P6, negative_zero) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__avx2_rr1_p6);
  }

  TEST(EXPM1MINUS__AVX2_RR1_P6, negative_saturation) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__avx2_rr1_p6, -1.0f);
  }

  TEST(EXPM1MINUS__AVX2_RR1_P6, nan) {
    TEST_REQUIRES_X86_AVX2;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__avx2_rr1_p6);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX_RR2_LUT4_P4_PERM, negative_zero) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__avx_rr2_lut4_p4_perm);
  }

  TEST(EXPM1MINUS__AVX_RR2_LUT4_P4_PERM, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__avx_rr2_lut4_p4_perm, -1.0f);
  }

  TEST(EXPM1MINUS__AVX_RR2_LUT4_P4_PERM, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__avx_rr2_lut4_p4_perm);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX_RR2_LUT16_P3, negative_zero) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__avx_rr2_lut16_p3);
  }

  TEST(EXPM1MINUS__AVX_RR2_LUT16_P3, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__avx_rr2_lut16_p3, -1.0f);
  }

  TEST(EXPM1MINUS__AVX_RR2_LUT16_P3, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__avx_rr2_lut16_p3);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__AVX_RR2_P6, negative_zero) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__avx_rr2_p6);
  }

  TEST(EXPM1MINUS__AVX_RR2_P6, negative_saturation) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__avx_rr2_p6, -1.0f);
  }

  TEST(EXPM1MINUS__AVX_RR2_P6, nan) {
    TEST_REQUIRES_X86_AVX;

    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__avx_rr2_p6);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__SSE2_RR2_LUT16_P3, negative_zero) {
    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__sse2_rr2_lut16_p3);
  }

  TEST(EXPM1MINUS__SSE2_RR2_LUT16_P3, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__sse2_rr2_lut16_p3, -1.0f);
  }

  TEST(EXPM1MINUS__SSE2_RR2_LUT16_P3, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__sse2_rr2_lut16_p3);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(EXPM1MINUS__SSE2_RR2_P6, negative_zero) {
    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__sse2_rr2_p6);
  }

  TEST(EXPM1MINUS__SSE2_RR2_P6, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__sse2_rr2_p6, -1.0f);
  }

  TEST(EXPM1MINUS__SSE2_RR2_P6, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__sse2_rr2_p6);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_ANDNOT, negative_zero) {
    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_ANDNOT, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot, -1.0f);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_ANDNOT, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_MAX, negative_zero) {
    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_max);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_MAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_max, -1.0f);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_LUT16_P3_MAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_max);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_ANDNOT, negative_zero) {
    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__wasmsimd_rr2_p6_andnot);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_ANDNOT, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__wasmsimd_rr2_p6_andnot, -1.0f);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_ANDNOT, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__wasmsimd_rr2_p6_andnot);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_MAX, negative_zero) {
    MathEvaluationTester()
      .input_value(-0.0f)
      .TestOutputMatchZero(xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_MAX, negative_saturation) {
    MathEvaluationTester()
      .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
      .TestOutputMatchReference(xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max, -1.0f);
  }

  TEST(EXPM1MINUS__WASMSIMD_RR2_P6_MAX, nan) {
    MathEvaluationTester()
      .TestNaN(xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(EXPM1MINUS__SCALAR_RR2_LUT4_P4, negative_zero) {
  MathEvaluationTester()
    .input_value(-0.0f)
    .TestOutputMatchZero(xnn_math_f32_expm1minus__scalar_rr2_lut4_p4);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT4_P4, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
    .TestOutputMatchReference(xnn_math_f32_expm1minus__scalar_rr2_lut4_p4, -1.0f);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT4_P4, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_expm1minus__scalar_rr2_lut4_p4);
}


TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P3, negative_zero) {
  MathEvaluationTester()
    .input_value(-0.0f)
    .TestOutputMatchZero(xnn_math_f32_expm1minus__scalar_rr2_lut8_p3);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P3, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
    .TestOutputMatchReference(xnn_math_f32_expm1minus__scalar_rr2_lut8_p3, -1.0f);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P3, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_expm1minus__scalar_rr2_lut8_p3);
}


TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P4, negative_zero) {
  MathEvaluationTester()
    .input_value(-0.0f)
    .TestOutputMatchZero(xnn_math_f32_expm1minus__scalar_rr2_lut8_p4);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P4, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
    .TestOutputMatchReference(xnn_math_f32_expm1minus__scalar_rr2_lut8_p4, -1.0f);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT8_P4, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_expm1minus__scalar_rr2_lut8_p4);
}


TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P3, negative_zero) {
  MathEvaluationTester()
    .input_value(-0.0f)
    .TestOutputMatchZero(xnn_math_f32_expm1minus__scalar_rr2_lut16_p3);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P3, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
    .TestOutputMatchReference(xnn_math_f32_expm1minus__scalar_rr2_lut16_p3, -1.0f);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P3, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_expm1minus__scalar_rr2_lut16_p3);
}


TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P4, negative_zero) {
  MathEvaluationTester()
    .input_value(-0.0f)
    .TestOutputMatchZero(xnn_math_f32_expm1minus__scalar_rr2_lut16_p4);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P4, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
    .TestOutputMatchReference(xnn_math_f32_expm1minus__scalar_rr2_lut16_p4, -1.0f);
}

TEST(EXPM1MINUS__SCALAR_RR2_LUT16_P4, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_expm1minus__scalar_rr2_lut16_p4);
}


TEST(EXPM1MINUS__SCALAR_RR2_P5, negative_zero) {
  MathEvaluationTester()
    .input_value(-0.0f)
    .TestOutputMatchZero(xnn_math_f32_expm1minus__scalar_rr2_p5);
}

TEST(EXPM1MINUS__SCALAR_RR2_P5, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
    .TestOutputMatchReference(xnn_math_f32_expm1minus__scalar_rr2_p5, -1.0f);
}

TEST(EXPM1MINUS__SCALAR_RR2_P5, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_expm1minus__scalar_rr2_p5);
}


TEST(EXPM1MINUS__SCALAR_RR2_P6, negative_zero) {
  MathEvaluationTester()
    .input_value(-0.0f)
    .TestOutputMatchZero(xnn_math_f32_expm1minus__scalar_rr2_p6);
}

TEST(EXPM1MINUS__SCALAR_RR2_P6, negative_saturation) {
  MathEvaluationTester()
    .input_range(-std::numeric_limits<float>::infinity(), -0x1.154246p+4f)
    .TestOutputMatchReference(xnn_math_f32_expm1minus__scalar_rr2_p6, -1.0f);
}

TEST(EXPM1MINUS__SCALAR_RR2_P6, nan) {
  MathEvaluationTester()
    .TestNaN(xnn_math_f32_expm1minus__scalar_rr2_p6);
}
