// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>
#include <cstdlib>

#include <gtest/gtest.h>

#include <xnnpack/common.h>

#include <xnnpack/requantization-stubs.h>
#include "requantization-tester.h"


/*
 * Precise scalar implementation using unsigned 32-bit arithmetics.
 */

TEST(PRECISE__SCALAR_UNSIGNED32, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .s(s)
      .TestExactDivideByPO2(xnn_requantize_precise__scalar_unsigned32);
  }
}

TEST(PRECISE__SCALAR_UNSIGNED32, exact_divide_by_po2_with_zero_point) {
  for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_up) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingUp(xnn_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_down) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingDown(xnn_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_away) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingAway(xnn_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED32, special_cases) {
  RequantizationTester()
    .TestSpecialCases(xnn_requantize_precise__scalar_unsigned32);
}

TEST(PRECISE__SCALAR_UNSIGNED32, random_cases) {
  RequantizationTester()
    .iterations(100)
    .TestRandomCasesPrecise(xnn_requantize_precise__scalar_unsigned32);
}


/*
 * Precise scalar implementation using unsigned 64-bit arithmetics.
 */

TEST(PRECISE__SCALAR_UNSIGNED64, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .s(s)
      .TestExactDivideByPO2(xnn_requantize_precise__scalar_unsigned64);
  }
}

TEST(PRECISE__SCALAR_UNSIGNED64, exact_divide_by_po2_with_zero_point) {
  for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_up) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingUp(xnn_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_down) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingDown(xnn_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_away) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingAway(xnn_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED64, special_cases) {
  RequantizationTester()
    .TestSpecialCases(xnn_requantize_precise__scalar_unsigned64);
}

TEST(PRECISE__SCALAR_UNSIGNED64, random_cases) {
  RequantizationTester()
    .iterations(100)
    .TestRandomCasesPrecise(xnn_requantize_precise__scalar_unsigned64);
}


/*
 * Precise scalar implementation using signed 64-bit arithmetics.
 */

TEST(PRECISE__SCALAR_SIGNED64, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .s(s)
      .TestExactDivideByPO2(xnn_requantize_precise__scalar_signed64);
  }
}

TEST(PRECISE__SCALAR_SIGNED64, exact_divide_by_po2_with_zero_point) {
  for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_precise__scalar_signed64);
    }
  }
}

TEST(PRECISE__SCALAR_SIGNED64, divide_by_po2_with_rounding_up) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingUp(xnn_requantize_precise__scalar_signed64);
    }
  }
}

TEST(PRECISE__SCALAR_SIGNED64, divide_by_po2_with_rounding_down) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingDown(xnn_requantize_precise__scalar_signed64);
    }
  }
}

TEST(PRECISE__SCALAR_SIGNED64, divide_by_po2_with_rounding_away) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingAway(xnn_requantize_precise__scalar_signed64);
    }
  }
}

TEST(PRECISE__SCALAR_SIGNED64, special_cases) {
  RequantizationTester()
    .TestSpecialCases(xnn_requantize_precise__scalar_signed64);
}

TEST(PRECISE__SCALAR_SIGNED64, random_cases) {
  RequantizationTester()
    .iterations(100)
    .TestRandomCasesPrecise(xnn_requantize_precise__scalar_signed64);
}


/*
 * FP32-based scalar implementation using lrintf function.
 */

TEST(FP32__SCALAR_LRINTF, random_cases) {
  RequantizationTester()
    .iterations(1000)
    .TestRandomCasesApproximate(xnn_requantize_fp32__scalar_lrintf);
}


/*
 * FP32-based scalar implementation using magic trick for FP32->INT32 conversion.
 */

TEST(FP32__SCALAR_MAGIC, random_cases) {
  RequantizationTester()
    .iterations(1000)
    .TestRandomCasesApproximate(xnn_requantize_fp32__scalar_magic);
}


/*
 * Q31-based scalar implementation.
 */

TEST(Q31__SCALAR, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .s(s)
      .TestExactDivideByPO2(xnn_requantize_q31__scalar);
  }
}

TEST(Q31__SCALAR, exact_divide_by_po2_with_zero_point) {
  for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_q31__scalar);
    }
  }
}

TEST(Q31__SCALAR, divide_by_po2_with_rounding_up) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingUp(xnn_requantize_q31__scalar);
    }
  }
}

/* No rounding down Test - it fails because of upward bias in multiplication */

TEST(Q31__SCALAR, divide_by_po2_with_rounding_away) {
  for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .s(s)
        .TestDivideByPO2WithRoundingAway(xnn_requantize_q31__scalar);
    }
  }
}

TEST(Q31__SCALAR, special_cases) {
  RequantizationTester()
    .TestSpecialCases(xnn_requantize_q31__scalar);
}

TEST(Q31__SCALAR, random_cases) {
  RequantizationTester()
    .iterations(100)
    .TestRandomCasesApproximate(xnn_requantize_q31__scalar);
}

TEST(Q31__SCALAR, random_match_gemmlowp) {
  RequantizationTester()
    .iterations(100)
    .TestRandomCasesAgainstReference(xnn_requantize_q31__scalar, xnn_requantize_gemmlowp__scalar);
}


/*
 * Scalar implementation from gemmlowp.
 */

TEST(GEMMLOWP__SCALAR, random_cases) {
  RequantizationTester()
    .iterations(100)
    .TestRandomCasesApproximate(xnn_requantize_gemmlowp__scalar);
}


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM
  /*
   * Precise PSIMD implementation using unsigned 32-bit arithmetics.
   */

  TEST(PRECISE__PSIMD, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_precise__psimd);
    }
  }

  TEST(PRECISE__PSIMD, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_precise__psimd);
      }
    }
  }

  TEST(PRECISE__PSIMD, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_precise__psimd);
      }
    }
  }

  TEST(PRECISE__PSIMD, divide_by_po2_with_rounding_down) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_requantize_precise__psimd);
      }
    }
  }

  TEST(PRECISE__PSIMD, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_precise__psimd);
      }
    }
  }

  TEST(PRECISE__PSIMD, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_precise__psimd);
  }

  TEST(PRECISE__PSIMD, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesPrecise(xnn_requantize_precise__psimd);
  }


  /*
   * FP32-based PSIMD implementation using magic trick for FP32->INT32 conversion.
   */

  TEST(FP32__PSIMD, random_cases) {
    RequantizationTester()
      .iterations(1000)
      .TestRandomCasesApproximate(xnn_requantize_fp32__psimd);
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  /*
   * Precise SSE2 implementation using floating-point shuffle.
   */

  TEST(PRECISE__SSE2, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_precise__sse2);
    }
  }

  TEST(PRECISE__SSE2, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_precise__sse2);
      }
    }
  }

  TEST(PRECISE__SSE2, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_precise__sse2);
      }
    }
  }

  TEST(PRECISE__SSE2, divide_by_po2_with_rounding_down) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_requantize_precise__sse2);
      }
    }
  }

  TEST(PRECISE__SSE2, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_precise__sse2);
      }
    }
  }

  TEST(PRECISE__SSE2, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_precise__sse2);
  }

  TEST(PRECISE__SSE2, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesPrecise(xnn_requantize_precise__sse2);
  }


  /*
   * Precise SSSE3 implementation using floating-point shuffle.
   */

  TEST(PRECISE__SSSE3, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_precise__ssse3);
    }
  }

  TEST(PRECISE__SSSE3, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_precise__ssse3);
      }
    }
  }

  TEST(PRECISE__SSSE3, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_precise__ssse3);
      }
    }
  }

  TEST(PRECISE__SSSE3, divide_by_po2_with_rounding_down) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_requantize_precise__ssse3);
      }
    }
  }

  TEST(PRECISE__SSSE3, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_precise__ssse3);
      }
    }
  }

  TEST(PRECISE__SSSE3, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_precise__ssse3);
  }

  TEST(PRECISE__SSSE3, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesPrecise(xnn_requantize_precise__ssse3);
  }


  /*
   * Precise SSE4.1 implementation using static blend instruction.
   */

  TEST(PRECISE__SSE4, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_precise__sse4);
    }
  }

  TEST(PRECISE__SSE4, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_precise__sse4);
      }
    }
  }

  TEST(PRECISE__SSE4, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_precise__sse4);
      }
    }
  }

  TEST(PRECISE__SSE4, divide_by_po2_with_rounding_down) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_requantize_precise__sse4);
      }
    }
  }

  TEST(PRECISE__SSE4, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_precise__sse4);
      }
    }
  }

  TEST(PRECISE__SSE4, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_precise__sse4);
  }

  TEST(PRECISE__SSE4, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesPrecise(xnn_requantize_precise__sse4);
  }


  /*
   * FP32-based x86 SSE2 implementation.
   */

  TEST(FP32__SSE2, random_cases) {
    RequantizationTester()
      .iterations(1000)
      .TestRandomCasesApproximate(xnn_requantize_fp32__sse2);
  }


  /*
   * Q31-based x86 SSE2 implementation.
   */

  TEST(Q31__SSE2, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_q31__sse2);
    }
  }

  TEST(Q31__SSE2, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_q31__sse2);
      }
    }
  }

  TEST(Q31__SSE2, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_q31__sse2);
      }
    }
  }

  /* No rounding down Test - it fails because of upward bias in multiplication */

  TEST(Q31__SSE2, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_q31__sse2);
      }
    }
  }

  TEST(Q31__SSE2, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_q31__sse2);
  }

  TEST(Q31__SSE2, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesApproximate(xnn_requantize_q31__sse2);
  }

  TEST(Q31__SSE2, random_match_gemmlowp) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesAgainstReference(xnn_requantize_q31__sse2, xnn_requantize_gemmlowp__sse2);
  }


  /*
   * Q31-based x86 SSSE3 implementation.
   */

  TEST(Q31__SSSE3, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_q31__ssse3);
    }
  }

  TEST(Q31__SSSE3, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_q31__ssse3);
      }
    }
  }

  TEST(Q31__SSSE3, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_q31__ssse3);
      }
    }
  }

  /* No rounding down Test - it fails because of upward bias in multiplication */

  TEST(Q31__SSSE3, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_q31__ssse3);
      }
    }
  }

  TEST(Q31__SSSE3, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_q31__ssse3);
  }

  TEST(Q31__SSSE3, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesApproximate(xnn_requantize_q31__ssse3);
  }

  TEST(Q31__SSSE3, random_match_gemmlowp) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesAgainstReference(xnn_requantize_q31__ssse3, xnn_requantize_gemmlowp__ssse3);
  }


  /*
   * Q31-based x86 SSE4 implementation.
   */

  TEST(Q31__SSE4, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_q31__sse4);
    }
  }

  TEST(Q31__SSE4, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_q31__sse4);
      }
    }
  }

  TEST(Q31__SSE4, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_q31__sse4);
      }
    }
  }

  /* No rounding down Test - it fails because of upward bias in multiplication */

  TEST(Q31__SSE4, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_q31__sse4);
      }
    }
  }

  TEST(Q31__SSE4, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_q31__sse4);
  }

  TEST(Q31__SSE4, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesApproximate(xnn_requantize_q31__sse4);
  }

  TEST(Q31__SSE4, random_match_gemmlowp) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesAgainstReference(xnn_requantize_q31__sse4, xnn_requantize_gemmlowp__sse4);
  }


  /*
   * x86 SSE2 implementation from gemmlowp.
   */

  TEST(GEMMLOWP__SSE2, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_gemmlowp__sse2);
    }
  }

  TEST(GEMMLOWP__SSE2, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_gemmlowp__sse2);
      }
    }
  }

  TEST(GEMMLOWP__SSE2, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_gemmlowp__sse2);
      }
    }
  }

  /* No rounding down Test - it fails because of upward bias in multiplication */

  TEST(GEMMLOWP__SSE2, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_gemmlowp__sse2);
      }
    }
  }

  TEST(GEMMLOWP__SSE2, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_gemmlowp__sse2);
  }

  TEST(GEMMLOWP__SSE2, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesApproximate(xnn_requantize_gemmlowp__sse2);
  }


  /*
   * x86 SSSE3 implementation from gemmlowp.
   */

  TEST(GEMMLOWP__SSSE3, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_gemmlowp__ssse3);
    }
  }

  TEST(GEMMLOWP__SSSE3, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_gemmlowp__ssse3);
      }
    }
  }

  TEST(GEMMLOWP__SSSE3, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_gemmlowp__ssse3);
      }
    }
  }

  /* No rounding down Test - it fails because of upward bias in multiplication */

  TEST(GEMMLOWP__SSSE3, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_gemmlowp__ssse3);
      }
    }
  }

  TEST(GEMMLOWP__SSSE3, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_gemmlowp__ssse3);
  }

  TEST(GEMMLOWP__SSSE3, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesApproximate(xnn_requantize_gemmlowp__ssse3);
  }


  /*
   * x86 SSE4 implementation from gemmlowp.
   */

  TEST(GEMMLOWP__SSE4, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_gemmlowp__sse4);
    }
  }

  TEST(GEMMLOWP__SSE4, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_gemmlowp__sse4);
      }
    }
  }

  TEST(GEMMLOWP__SSE4, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_gemmlowp__sse4);
      }
    }
  }

  /* No rounding down Test - it fails because of upward bias in multiplication */

  TEST(GEMMLOWP__SSE4, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_gemmlowp__sse4);
      }
    }
  }

  TEST(GEMMLOWP__SSE4, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_gemmlowp__sse4);
  }

  TEST(GEMMLOWP__SSE4, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesApproximate(xnn_requantize_gemmlowp__sse4);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  /*
   * Precise ARM NEON implementation.
   */

  TEST(PRECISE__NEON, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_precise__neon);
    }
  }

  TEST(PRECISE__NEON, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_precise__neon);
      }
    }
  }

  TEST(PRECISE__NEON, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_precise__neon);
      }
    }
  }

  TEST(PRECISE__NEON, divide_by_po2_with_rounding_down) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_requantize_precise__neon);
      }
    }
  }

  TEST(PRECISE__NEON, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_precise__neon);
      }
    }
  }

  TEST(PRECISE__NEON, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_precise__neon);
  }

  TEST(PRECISE__NEON, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesPrecise(xnn_requantize_precise__neon);
  }


  /*
   * FP32-based ARM NEON implementation.
   */

  TEST(FP32__NEON, random_cases) {
    RequantizationTester()
      .iterations(1000)
      .TestRandomCasesApproximate(xnn_requantize_fp32__neon);
  }


  /*
   * Q31-based ARM NEON implementation.
   */

  TEST(Q31__NEON, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .TestExactDivideByPO2(xnn_requantize_q31__neon);
    }
  }

  TEST(Q31__NEON, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = 1; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestExactDivideByPO2(xnn_requantize_q31__neon);
      }
    }
  }

  TEST(Q31__NEON, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_requantize_q31__neon);
      }
    }
  }

  /* No rounding down Test - it fails because of upward bias in multiplication */

  TEST(Q31__NEON, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .s(s)
          .TestDivideByPO2WithRoundingAway(xnn_requantize_q31__neon);
      }
    }
  }

  TEST(Q31__NEON, special_cases) {
    RequantizationTester()
      .TestSpecialCases(xnn_requantize_q31__neon);
  }

  TEST(Q31__NEON, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesApproximate(xnn_requantize_q31__neon);
  }

  TEST(Q31__NEON, random_match_gemmlowp) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesAgainstReference(xnn_requantize_q31__neon, xnn_requantize_gemmlowp__neon);
  }


  /*
   * ARM NEON implementation from gemmlowp.
   */

  TEST(GEMMLOWP__NEON, random_cases) {
    RequantizationTester()
      .iterations(100)
      .TestRandomCasesApproximate(xnn_requantize_gemmlowp__neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
