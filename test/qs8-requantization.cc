// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>
#include <cstdlib>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/requantization-stubs.h>
#include "requantization-tester.h"


/*
 * Round-to-nearest, ties away from zero, scalar implementation using unsigned 32-bit arithmetics.
 */

TEST(QS8_RNDNA__SCALAR_UNSIGNED32, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .s(s)
      .TestExactDivideByPO2(xnn_qs8_requantize_rndna__scalar_unsigned32);
  }
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED32, exact_divide_by_po2_with_zero_point) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndna__scalar_unsigned32);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_up) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndna__scalar_unsigned32);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_down) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndna__scalar_unsigned32);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_away) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingTiesAway(xnn_qs8_requantize_rndna__scalar_unsigned32);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED32, special_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .TestSpecialCases(xnn_qs8_requantize_rndna__scalar_unsigned32);
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED32, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(100)
    .TestRandomCasesRoundToNearestTiesAway(xnn_qs8_requantize_rndna__scalar_unsigned32);
}


/*
 * Round-to-nearest, ties away from zero, scalar implementation using unsigned 64-bit arithmetics.
 */

TEST(QS8_RNDNA__SCALAR_UNSIGNED64, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .s(s)
      .TestExactDivideByPO2(xnn_qs8_requantize_rndna__scalar_unsigned64);
  }
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED64, exact_divide_by_po2_with_zero_point) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndna__scalar_unsigned64);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_up) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndna__scalar_unsigned64);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_down) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndna__scalar_unsigned64);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_away) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingTiesAway(xnn_qs8_requantize_rndna__scalar_unsigned64);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED64, special_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .TestSpecialCases(xnn_qs8_requantize_rndna__scalar_unsigned64);
}

TEST(QS8_RNDNA__SCALAR_UNSIGNED64, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(100)
    .TestRandomCasesRoundToNearestTiesAway(xnn_qs8_requantize_rndna__scalar_unsigned64);
}


/*
 * Round-to-nearest, ties away from zero, scalar implementation using signed 64-bit arithmetics.
 */

TEST(QS8_RNDNA__SCALAR_SIGNED64, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .s(s)
      .TestExactDivideByPO2(xnn_qs8_requantize_rndna__scalar_signed64);
  }
}

TEST(QS8_RNDNA__SCALAR_SIGNED64, exact_divide_by_po2_with_zero_point) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndna__scalar_signed64);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_SIGNED64, divide_by_po2_with_rounding_up) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndna__scalar_signed64);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_SIGNED64, divide_by_po2_with_rounding_down) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndna__scalar_signed64);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_SIGNED64, divide_by_po2_with_rounding_away) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingTiesAway(xnn_qs8_requantize_rndna__scalar_signed64);
    }
  }
}

TEST(QS8_RNDNA__SCALAR_SIGNED64, special_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .TestSpecialCases(xnn_qs8_requantize_rndna__scalar_signed64);
}

TEST(QS8_RNDNA__SCALAR_SIGNED64, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(100)
    .TestRandomCasesRoundToNearestTiesAway(xnn_qs8_requantize_rndna__scalar_signed64);
}


/*
 * Round-to-nearest, ties up, scalar implementation using signed 64-bit arithmetics.
 */

TEST(QS8_RNDNU__SCALAR, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .s(s)
      .TestExactDivideByPO2(xnn_qs8_requantize_rndnu__scalar);
  }
}

TEST(QS8_RNDNU__SCALAR, exact_divide_by_po2_with_zero_point) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndnu__scalar);
    }
  }
}

TEST(QS8_RNDNU__SCALAR, divide_by_po2_with_rounding_up) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndnu__scalar);
    }
  }
}

TEST(QS8_RNDNU__SCALAR, divide_by_po2_with_rounding_down) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndnu__scalar);
    }
  }
}

TEST(QS8_RNDNU__SCALAR, divide_by_po2_with_rounding_away) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingTiesUp(xnn_qs8_requantize_rndnu__scalar);
    }
  }
}

TEST(QS8_RNDNU__SCALAR, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(100)
    .TestRandomCasesRoundToNearestTiesUp(xnn_qs8_requantize_rndnu__scalar);
}


/*
 * FP32-based scalar implementation using lrintf function.
 */

TEST(QS8_FP32__SCALAR_LRINTF, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(1000)
    .TestRandomCasesApproximate(xnn_qs8_requantize_fp32__scalar_lrintf);
}


/*
 * FP32-based scalar implementation using magic trick for FP32->INT32 conversion.
 */

TEST(QS8_FP32__SCALAR_FMAGIC, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(1000)
    .TestRandomCasesApproximate(xnn_qs8_requantize_fp32__scalar_fmagic);
}


/*
 * GEMMLOWP-equivalent scalar implementation.
 */

TEST(QS8_GEMMLOWP__SCALAR, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .s(s)
      .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__scalar);
  }
}

TEST(QS8_GEMMLOWP__SCALAR, exact_divide_by_po2_with_zero_point) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__scalar);
    }
  }
}

TEST(QS8_GEMMLOWP__SCALAR, divide_by_po2_with_rounding_up) {
  for (int32_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point++)
  {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_gemmlowp__scalar);
    }
  }
}

/* No rounding down test - it fails because of upward bias in multiplication */
/* No rounding away test - it fails because of upward bias in multiplication */

TEST(QS8_GEMMLOWP__SCALAR, special_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .TestSpecialCases(xnn_qs8_requantize_gemmlowp__scalar);
}

TEST(QS8_GEMMLOWP__SCALAR, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(100)
    .TestRandomCasesApproximate(xnn_qs8_requantize_gemmlowp__scalar);
}


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  /*
   * Round-to-nearest, ties away from zero, SSE2 implementation using floating-point shuffle.
   */

  TEST(QS8_RNDNA__SSE2, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndna__sse2);
    }
  }

  TEST(QS8_RNDNA__SSE2, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_rndna__sse2);
      }
    }
  }

  TEST(QS8_RNDNA__SSE2, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndna__sse2);
      }
    }
  }

  TEST(QS8_RNDNA__SSE2, divide_by_po2_with_rounding_down) {
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndna__sse2);
      }
    }
  }

  TEST(QS8_RNDNA__SSE2, divide_by_po2_with_rounding_away) {
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingTiesAway(xnn_qs8_requantize_rndna__sse2);
      }
    }
  }

  TEST(QS8_RNDNA__SSE2, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_rndna__sse2);
  }

  TEST(QS8_RNDNA__SSE2, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesRoundToNearestTiesAway(xnn_qs8_requantize_rndna__sse2);
  }


  /*
   * Round-to-nearest, ties away from zero, SSSE3 implementation using floating-point shuffle.
   */

  TEST(QS8_RNDNA__SSSE3, exact_divide_by_po2) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndna__ssse3);
    }
  }

  TEST(QS8_RNDNA__SSSE3, exact_divide_by_po2_with_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_rndna__ssse3);
      }
    }
  }

  TEST(QS8_RNDNA__SSSE3, divide_by_po2_with_rounding_up) {
    TEST_REQUIRES_X86_SSSE3;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndna__ssse3);
      }
    }
  }

  TEST(QS8_RNDNA__SSSE3, divide_by_po2_with_rounding_down) {
    TEST_REQUIRES_X86_SSSE3;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndna__ssse3);
      }
    }
  }

  TEST(QS8_RNDNA__SSSE3, divide_by_po2_with_rounding_away) {
    TEST_REQUIRES_X86_SSSE3;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingTiesAway(xnn_qs8_requantize_rndna__ssse3);
      }
    }
  }

  TEST(QS8_RNDNA__SSSE3, special_cases) {
    TEST_REQUIRES_X86_SSSE3;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_rndna__ssse3);
  }

  TEST(QS8_RNDNA__SSSE3, random_cases) {
    TEST_REQUIRES_X86_SSSE3;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesRoundToNearestTiesAway(xnn_qs8_requantize_rndna__ssse3);
  }


  /*
   * Round-to-nearest, ties away from zero, SSE4.1 implementation using static blend instruction.
   */

  TEST(QS8_RNDNA__SSE41, exact_divide_by_po2) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndna__sse41);
    }
  }

  TEST(QS8_RNDNA__SSE41, exact_divide_by_po2_with_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_rndna__sse41);
      }
    }
  }

  TEST(QS8_RNDNA__SSE41, divide_by_po2_with_rounding_up) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndna__sse41);
      }
    }
  }

  TEST(QS8_RNDNA__SSE41, divide_by_po2_with_rounding_down) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndna__sse41);
      }
    }
  }

  TEST(QS8_RNDNA__SSE41, divide_by_po2_with_rounding_away) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingTiesAway(xnn_qs8_requantize_rndna__sse41);
      }
    }
  }

  TEST(QS8_RNDNA__SSE41, special_cases) {
    TEST_REQUIRES_X86_SSE41;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_rndna__sse41);
  }

  TEST(QS8_RNDNA__SSE41, random_cases) {
    TEST_REQUIRES_X86_SSE41;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesRoundToNearestTiesAway(xnn_qs8_requantize_rndna__sse41);
  }


  /*
   * Round-to-nearest, ties up, SSE4.1 implementation using arithmetic shift right.
   */

  TEST(QS8_RNDNU__SSE4_SRA, exact_divide_by_po2) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndnu__sse41_sra);
    }
  }

  TEST(QS8_RNDNU__SSE4_SRA, exact_divide_by_po2_with_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_rndnu__sse41_sra);
      }
    }
  }

  TEST(QS8_RNDNU__SSE4_SRA, divide_by_po2_with_rounding_up) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndnu__sse41_sra);
      }
    }
  }

  TEST(QS8_RNDNU__SSE4_SRA, divide_by_po2_with_rounding_down) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndnu__sse41_sra);
      }
    }
  }

  TEST(QS8_RNDNU__SSE4_SRA, divide_by_po2_with_rounding_away) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingTiesUp(xnn_qs8_requantize_rndnu__sse41_sra);
      }
    }
  }

  TEST(QS8_RNDNU__SSE4_SRA, random_cases) {
    TEST_REQUIRES_X86_SSE41;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesRoundToNearestTiesUp(xnn_qs8_requantize_rndnu__sse41_sra);
  }


  /*
   * Round-to-nearest, ties up, SSE4.1 implementation using logical shift right.
   */

  TEST(QS8_RNDNU__SSE4_SRL, exact_divide_by_po2) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndnu__sse41_srl);
    }
  }

  TEST(QS8_RNDNU__SSE4_SRL, exact_divide_by_po2_with_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_rndnu__sse41_srl);
      }
    }
  }

  TEST(QS8_RNDNU__SSE4_SRL, divide_by_po2_with_rounding_up) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndnu__sse41_srl);
      }
    }
  }

  TEST(QS8_RNDNU__SSE4_SRL, divide_by_po2_with_rounding_down) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndnu__sse41_srl);
      }
    }
  }

  TEST(QS8_RNDNU__SSE4_SRL, divide_by_po2_with_rounding_away) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingTiesUp(xnn_qs8_requantize_rndnu__sse41_srl);
      }
    }
  }

  TEST(QS8_RNDNU__SSE4_SRL, random_cases) {
    TEST_REQUIRES_X86_SSE41;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesRoundToNearestTiesUp(xnn_qs8_requantize_rndnu__sse41_srl);
  }


  /*
   * FP32-based x86 SSE2 implementation.
   */

  TEST(QS8_FP32__SSE2, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(1000)
      .TestRandomCasesApproximate(xnn_qs8_requantize_fp32__sse2);
  }


  /*
   * FP32-based x86 SSE4 implementation.
   */

  TEST(QS8_FP32__SSE41, random_cases) {
    TEST_REQUIRES_X86_SSE41;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(1000)
      .TestRandomCasesApproximate(xnn_qs8_requantize_fp32__sse41);
  }


  /*
   * GEMMLOWP-equivalent x86 SSE2 implementation.
   */

  TEST(QS8_GEMMLOWP__SSE2, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__sse2);
    }
  }

  TEST(QS8_GEMMLOWP__SSE2, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__sse2);
      }
    }
  }

  TEST(QS8_GEMMLOWP__SSE2, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_gemmlowp__sse2);
      }
    }
  }

  /* No rounding down test - it fails because of upward bias in multiplication */
  /* No rounding away test - it fails because of upward bias in multiplication */

  TEST(QS8_GEMMLOWP__SSE2, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_gemmlowp__sse2);
  }

  TEST(QS8_GEMMLOWP__SSE2, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesApproximate(xnn_qs8_requantize_gemmlowp__sse2);
  }


  /*
   * GEMMLOWP-equivalent x86 SSSE3 implementation.
   */

  TEST(QS8_GEMMLOWP__SSSE3, exact_divide_by_po2) {
    TEST_REQUIRES_X86_SSSE3;
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__ssse3);
    }
  }

  TEST(QS8_GEMMLOWP__SSSE3, exact_divide_by_po2_with_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__ssse3);
      }
    }
  }

  TEST(QS8_GEMMLOWP__SSSE3, divide_by_po2_with_rounding_up) {
    TEST_REQUIRES_X86_SSSE3;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_gemmlowp__ssse3);
      }
    }
  }

  /* No rounding down test - it fails because of upward bias in multiplication */
  /* No rounding away test - it fails because of upward bias in multiplication */

  TEST(QS8_GEMMLOWP__SSSE3, special_cases) {
    TEST_REQUIRES_X86_SSSE3;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_gemmlowp__ssse3);
  }

  TEST(QS8_GEMMLOWP__SSSE3, random_cases) {
    TEST_REQUIRES_X86_SSSE3;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesApproximate(xnn_qs8_requantize_gemmlowp__ssse3);
  }


  /*
   * GEMMLOWP-equivalent x86 SSE4 implementation.
   */

  TEST(QS8_GEMMLOWP__SSE41, exact_divide_by_po2) {
    TEST_REQUIRES_X86_SSE41;
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__sse41);
    }
  }

  TEST(QS8_GEMMLOWP__SSE41, exact_divide_by_po2_with_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__sse41);
      }
    }
  }

  TEST(QS8_GEMMLOWP__SSE41, divide_by_po2_with_rounding_up) {
    TEST_REQUIRES_X86_SSE41;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_gemmlowp__sse41);
      }
    }
  }

  /* No rounding down test - it fails because of upward bias in multiplication */
  /* No rounding away test - it fails because of upward bias in multiplication */

  TEST(QS8_GEMMLOWP__SSE41, special_cases) {
    TEST_REQUIRES_X86_SSE41;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_gemmlowp__sse41);
  }

  TEST(QS8_GEMMLOWP__SSE41, random_cases) {
    TEST_REQUIRES_X86_SSE41;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesApproximate(xnn_qs8_requantize_gemmlowp__sse41);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  /*
   * Round-to-nearest, ties away from zero, ARM NEON implementation.
   */

  TEST(QS8_RNDNA__NEON, exact_divide_by_po2) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestExactDivideByPO2(xnn_qs8_requantize_rndna__neon);
    }
  }

  TEST(QS8_RNDNA__NEON, exact_divide_by_po2_with_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_rndna__neon);
      }
    }
  }

  TEST(QS8_RNDNA__NEON, divide_by_po2_with_rounding_up) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndna__neon);
      }
    }
  }

  TEST(QS8_RNDNA__NEON, divide_by_po2_with_rounding_down) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndna__neon);
      }
    }
  }

  TEST(QS8_RNDNA__NEON, divide_by_po2_with_rounding_away) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingTiesAway(xnn_qs8_requantize_rndna__neon);
      }
    }
  }

  TEST(QS8_RNDNA__NEON, special_cases) {
    TEST_REQUIRES_ARM_NEON;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_rndna__neon);
  }

  TEST(QS8_RNDNA__NEON, random_cases) {
    TEST_REQUIRES_ARM_NEON;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesRoundToNearestTiesAway(xnn_qs8_requantize_rndna__neon);
  }


  /*
   * Round-to-nearest, ties up, ARM NEON implementation using extended multiplication.
   */

  TEST(QS8_RNDNU__NEON_MULL, exact_divide_by_po2) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndnu__neon_mull);
    }
  }

  TEST(QS8_RNDNU__NEON_MULL, exact_divide_by_po2_with_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_rndnu__neon_mull);
      }
    }
  }

  TEST(QS8_RNDNU__NEON_MULL, divide_by_po2_with_rounding_up) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndnu__neon_mull);
      }
    }
  }

  TEST(QS8_RNDNU__NEON_MULL, divide_by_po2_with_rounding_down) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndnu__neon_mull);
      }
    }
  }

  TEST(QS8_RNDNU__NEON_MULL, divide_by_po2_with_rounding_away) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingTiesUp(xnn_qs8_requantize_rndnu__neon_mull);
      }
    }
  }

  TEST(QS8_RNDNU__NEON_MULL, random_cases) {
    TEST_REQUIRES_ARM_NEON;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesRoundToNearestTiesUp(xnn_qs8_requantize_rndnu__neon_mull);
  }


  /*
   * Round-to-nearest, ties up, ARM NEON implementation using Q31 multiplication.
   */

  TEST(QS8_RNDNU__NEON_QDMULH, exact_divide_by_po2) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_rndnu__neon_qdmulh);
    }
  }

  TEST(QS8_RNDNU__NEON_QDMULH, exact_divide_by_po2_with_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_rndnu__neon_qdmulh);
      }
    }
  }

  TEST(QS8_RNDNU__NEON_QDMULH, divide_by_po2_with_rounding_up) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_rndnu__neon_qdmulh);
      }
    }
  }

  TEST(QS8_RNDNU__NEON_QDMULH, divide_by_po2_with_rounding_down) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_rndnu__neon_qdmulh);
      }
    }
  }

  TEST(QS8_RNDNU__NEON_QDMULH, divide_by_po2_with_rounding_away) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingTiesUp(xnn_qs8_requantize_rndnu__neon_qdmulh);
      }
    }
  }

  TEST(QS8_RNDNU__NEON_QDMULH, random_cases) {
    TEST_REQUIRES_ARM_NEON;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesRoundToNearestTiesUp(xnn_qs8_requantize_rndnu__neon_qdmulh);
  }


  /*
   * FP32-based ARM NEON implementation.
   */

  TEST(QS8_FP32__NEON, random_cases) {
    TEST_REQUIRES_ARM_NEON;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(1000)
      .TestRandomCasesApproximate(xnn_qs8_requantize_fp32__neon);
  }


  /*
   * GEMMLOWP-equivalent ARM NEON implementation.
   */

  TEST(QS8_GEMMLOWP__NEON, exact_divide_by_po2) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__neon);
    }
  }

  TEST(QS8_GEMMLOWP__NEON, exact_divide_by_po2_with_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__neon);
      }
    }
  }

  TEST(QS8_GEMMLOWP__NEON, divide_by_po2_with_rounding_up) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_gemmlowp__neon);
      }
    }
  }

  /* No rounding down test - it fails because of upward bias in multiplication */
  /* No rounding away test - it fails because of upward bias in multiplication */

  TEST(QS8_GEMMLOWP__NEON, special_cases) {
    TEST_REQUIRES_ARM_NEON;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_gemmlowp__neon);
  }

  TEST(QS8_GEMMLOWP__NEON, random_cases) {
    TEST_REQUIRES_ARM_NEON;
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesApproximate(xnn_qs8_requantize_gemmlowp__neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD
  /*
   * FP32-based WAsm SIMD implementation.
   */

  TEST(QS8_FP32__WASMSIMD, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(1000)
      .TestRandomCasesApproximate(xnn_qs8_requantize_fp32__wasmsimd);
  }

  /*
   * GEMMLOWP-equivalent WAsm SIMD implementation.
   */

  TEST(QS8_GEMMLOWP__WASMSIMD, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__wasmsimd);
    }
  }

  TEST(QS8_GEMMLOWP__WASMSIMD, exact_divide_by_po2_with_zero_point) {
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestExactDivideByPO2(xnn_qs8_requantize_gemmlowp__wasmsimd);
      }
    }
  }

  TEST(QS8_GEMMLOWP__WASMSIMD, divide_by_po2_with_rounding_up) {
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      for (uint32_t s = 1; s < 32; s++) {
        RequantizationTester()
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .s(s)
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_gemmlowp__wasmsimd);
      }
    }
  }

  /* No rounding down test - it fails because of upward bias in multiplication */
  /* No rounding away test - it fails because of upward bias in multiplication */

  TEST(QS8_GEMMLOWP__WASMSIMD, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_gemmlowp__wasmsimd);
  }

  TEST(QS8_GEMMLOWP__WASMSIMD, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesApproximate(xnn_qs8_requantize_gemmlowp__wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD
