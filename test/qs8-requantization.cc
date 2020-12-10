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

#include <xnnpack/requantization-stubs.h>
#include "requantization-tester.h"


/*
 * Precise scalar implementation using unsigned 32-bit arithmetics.
 */

TEST(QS8_PRECISE__SCALAR_UNSIGNED32, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .s(s)
      .TestExactDivideByPO2(xnn_qs8_requantize_precise__scalar_unsigned32);
  }
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED32, exact_divide_by_po2_with_zero_point) {
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
        .TestExactDivideByPO2(xnn_qs8_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_up) {
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
        .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_down) {
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
        .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_away) {
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
        .TestDivideByPO2WithRoundingAway(xnn_qs8_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED32, special_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .TestSpecialCases(xnn_qs8_requantize_precise__scalar_unsigned32);
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED32, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(100)
    .TestRandomCasesPrecise(xnn_qs8_requantize_precise__scalar_unsigned32);
}


/*
 * Precise scalar implementation using unsigned 64-bit arithmetics.
 */

TEST(QS8_PRECISE__SCALAR_UNSIGNED64, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .s(s)
      .TestExactDivideByPO2(xnn_qs8_requantize_precise__scalar_unsigned64);
  }
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED64, exact_divide_by_po2_with_zero_point) {
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
        .TestExactDivideByPO2(xnn_qs8_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_up) {
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
        .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_down) {
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
        .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_away) {
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
        .TestDivideByPO2WithRoundingAway(xnn_qs8_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED64, special_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .TestSpecialCases(xnn_qs8_requantize_precise__scalar_unsigned64);
}

TEST(QS8_PRECISE__SCALAR_UNSIGNED64, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(100)
    .TestRandomCasesPrecise(xnn_qs8_requantize_precise__scalar_unsigned64);
}


/*
 * Precise scalar implementation using signed 64-bit arithmetics.
 */

TEST(QS8_PRECISE__SCALAR_SIGNED64, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .s(s)
      .TestExactDivideByPO2(xnn_qs8_requantize_precise__scalar_signed64);
  }
}

TEST(QS8_PRECISE__SCALAR_SIGNED64, exact_divide_by_po2_with_zero_point) {
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
        .TestExactDivideByPO2(xnn_qs8_requantize_precise__scalar_signed64);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_SIGNED64, divide_by_po2_with_rounding_up) {
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
        .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_precise__scalar_signed64);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_SIGNED64, divide_by_po2_with_rounding_down) {
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
        .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_precise__scalar_signed64);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_SIGNED64, divide_by_po2_with_rounding_away) {
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
        .TestDivideByPO2WithRoundingAway(xnn_qs8_requantize_precise__scalar_signed64);
    }
  }
}

TEST(QS8_PRECISE__SCALAR_SIGNED64, special_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .TestSpecialCases(xnn_qs8_requantize_precise__scalar_signed64);
}

TEST(QS8_PRECISE__SCALAR_SIGNED64, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(100)
    .TestRandomCasesPrecise(xnn_qs8_requantize_precise__scalar_signed64);
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

TEST(QS8_FP32__SCALAR_MAGIC, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(1000)
    .TestRandomCasesApproximate(xnn_qs8_requantize_fp32__scalar_magic);
}


/*
 * Q31-based scalar implementation.
 */

TEST(QS8_Q31__SCALAR, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .s(s)
      .TestExactDivideByPO2(xnn_qs8_requantize_q31__scalar);
  }
}

TEST(QS8_Q31__SCALAR, exact_divide_by_po2_with_zero_point) {
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
        .TestExactDivideByPO2(xnn_qs8_requantize_q31__scalar);
    }
  }
}

TEST(QS8_Q31__SCALAR, divide_by_po2_with_rounding_up) {
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
        .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_q31__scalar);
    }
  }
}

/* No rounding down test - it fails because of upward bias in multiplication */
/* No rounding away test - it fails because of upward bias in multiplication */

TEST(QS8_Q31__SCALAR, special_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .TestSpecialCases(xnn_qs8_requantize_q31__scalar);
}

TEST(QS8_Q31__SCALAR, random_cases) {
  RequantizationTester()
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .iterations(100)
    .TestRandomCasesApproximate(xnn_qs8_requantize_q31__scalar);
}


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  /*
   * Precise SSE2 implementation using floating-point shuffle.
   */

  TEST(QS8_PRECISE__SSE2, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_precise__sse2);
    }
  }

  TEST(QS8_PRECISE__SSE2, exact_divide_by_po2_with_zero_point) {
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
          .TestExactDivideByPO2(xnn_qs8_requantize_precise__sse2);
      }
    }
  }

  TEST(QS8_PRECISE__SSE2, divide_by_po2_with_rounding_up) {
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
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_precise__sse2);
      }
    }
  }

  TEST(QS8_PRECISE__SSE2, divide_by_po2_with_rounding_down) {
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
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_precise__sse2);
      }
    }
  }

  TEST(QS8_PRECISE__SSE2, divide_by_po2_with_rounding_away) {
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
          .TestDivideByPO2WithRoundingAway(xnn_qs8_requantize_precise__sse2);
      }
    }
  }

  TEST(QS8_PRECISE__SSE2, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_precise__sse2);
  }

  TEST(QS8_PRECISE__SSE2, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesPrecise(xnn_qs8_requantize_precise__sse2);
  }


  /*
   * Precise SSSE3 implementation using floating-point shuffle.
   */

  TEST(QS8_PRECISE__SSSE3, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_precise__ssse3);
    }
  }

  TEST(QS8_PRECISE__SSSE3, exact_divide_by_po2_with_zero_point) {
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
          .TestExactDivideByPO2(xnn_qs8_requantize_precise__ssse3);
      }
    }
  }

  TEST(QS8_PRECISE__SSSE3, divide_by_po2_with_rounding_up) {
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
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_precise__ssse3);
      }
    }
  }

  TEST(QS8_PRECISE__SSSE3, divide_by_po2_with_rounding_down) {
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
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_precise__ssse3);
      }
    }
  }

  TEST(QS8_PRECISE__SSSE3, divide_by_po2_with_rounding_away) {
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
          .TestDivideByPO2WithRoundingAway(xnn_qs8_requantize_precise__ssse3);
      }
    }
  }

  TEST(QS8_PRECISE__SSSE3, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_precise__ssse3);
  }

  TEST(QS8_PRECISE__SSSE3, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesPrecise(xnn_qs8_requantize_precise__ssse3);
  }


  /*
   * Precise SSE4.1 implementation using static blend instruction.
   */

  TEST(QS8_PRECISE__SSE4, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_precise__sse4);
    }
  }

  TEST(QS8_PRECISE__SSE4, exact_divide_by_po2_with_zero_point) {
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
          .TestExactDivideByPO2(xnn_qs8_requantize_precise__sse4);
      }
    }
  }

  TEST(QS8_PRECISE__SSE4, divide_by_po2_with_rounding_up) {
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
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_precise__sse4);
      }
    }
  }

  TEST(QS8_PRECISE__SSE4, divide_by_po2_with_rounding_down) {
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
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_precise__sse4);
      }
    }
  }

  TEST(QS8_PRECISE__SSE4, divide_by_po2_with_rounding_away) {
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
          .TestDivideByPO2WithRoundingAway(xnn_qs8_requantize_precise__sse4);
      }
    }
  }

  TEST(QS8_PRECISE__SSE4, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_precise__sse4);
  }

  TEST(QS8_PRECISE__SSE4, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesPrecise(xnn_qs8_requantize_precise__sse4);
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

  TEST(QS8_FP32__SSE4, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(1000)
      .TestRandomCasesApproximate(xnn_qs8_requantize_fp32__sse4);
  }


  /*
   * Q31-based x86 SSE2 implementation.
   */

  TEST(QS8_Q31__SSE2, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_q31__sse2);
    }
  }

  TEST(QS8_Q31__SSE2, exact_divide_by_po2_with_zero_point) {
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
          .TestExactDivideByPO2(xnn_qs8_requantize_q31__sse2);
      }
    }
  }

  TEST(QS8_Q31__SSE2, divide_by_po2_with_rounding_up) {
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
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_q31__sse2);
      }
    }
  }

  /* No rounding down test - it fails because of upward bias in multiplication */
  /* No rounding away test - it fails because of upward bias in multiplication */

  TEST(QS8_Q31__SSE2, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_q31__sse2);
  }

  TEST(QS8_Q31__SSE2, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesApproximate(xnn_qs8_requantize_q31__sse2);
  }


  /*
   * Q31-based x86 SSSE3 implementation.
   */

  TEST(QS8_Q31__SSSE3, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_q31__ssse3);
    }
  }

  TEST(QS8_Q31__SSSE3, exact_divide_by_po2_with_zero_point) {
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
          .TestExactDivideByPO2(xnn_qs8_requantize_q31__ssse3);
      }
    }
  }

  TEST(QS8_Q31__SSSE3, divide_by_po2_with_rounding_up) {
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
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_q31__ssse3);
      }
    }
  }

  /* No rounding down test - it fails because of upward bias in multiplication */
  /* No rounding away test - it fails because of upward bias in multiplication */

  TEST(QS8_Q31__SSSE3, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_q31__ssse3);
  }

  TEST(QS8_Q31__SSSE3, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesApproximate(xnn_qs8_requantize_q31__ssse3);
  }


  /*
   * Q31-based x86 SSE4 implementation.
   */

  TEST(QS8_Q31__SSE4, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_q31__sse4);
    }
  }

  TEST(QS8_Q31__SSE4, exact_divide_by_po2_with_zero_point) {
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
          .TestExactDivideByPO2(xnn_qs8_requantize_q31__sse4);
      }
    }
  }

  TEST(QS8_Q31__SSE4, divide_by_po2_with_rounding_up) {
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
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_q31__sse4);
      }
    }
  }

  /* No rounding down test - it fails because of upward bias in multiplication */
  /* No rounding away test - it fails because of upward bias in multiplication */

  TEST(QS8_Q31__SSE4, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_q31__sse4);
  }

  TEST(QS8_Q31__SSE4, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesApproximate(xnn_qs8_requantize_q31__sse4);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  /*
   * Precise ARM NEON implementation.
   */

  TEST(QS8_PRECISE__NEON, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .s(s)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .TestExactDivideByPO2(xnn_qs8_requantize_precise__neon);
    }
  }

  TEST(QS8_PRECISE__NEON, exact_divide_by_po2_with_zero_point) {
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
          .TestExactDivideByPO2(xnn_qs8_requantize_precise__neon);
      }
    }
  }

  TEST(QS8_PRECISE__NEON, divide_by_po2_with_rounding_up) {
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
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_precise__neon);
      }
    }
  }

  TEST(QS8_PRECISE__NEON, divide_by_po2_with_rounding_down) {
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
          .TestDivideByPO2WithRoundingDown(xnn_qs8_requantize_precise__neon);
      }
    }
  }

  TEST(QS8_PRECISE__NEON, divide_by_po2_with_rounding_away) {
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
          .TestDivideByPO2WithRoundingAway(xnn_qs8_requantize_precise__neon);
      }
    }
  }

  TEST(QS8_PRECISE__NEON, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_precise__neon);
  }

  TEST(QS8_PRECISE__NEON, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesPrecise(xnn_qs8_requantize_precise__neon);
  }


  /*
   * FP32-based ARM NEON implementation.
   */

  TEST(QS8_FP32__NEON, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(1000)
      .TestRandomCasesApproximate(xnn_qs8_requantize_fp32__neon);
  }


  /*
   * Q31-based ARM NEON implementation.
   */

  TEST(QS8_Q31__NEON, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_q31__neon);
    }
  }

  TEST(QS8_Q31__NEON, exact_divide_by_po2_with_zero_point) {
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
          .TestExactDivideByPO2(xnn_qs8_requantize_q31__neon);
      }
    }
  }

  TEST(QS8_Q31__NEON, divide_by_po2_with_rounding_up) {
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
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_q31__neon);
      }
    }
  }

  /* No rounding down test - it fails because of upward bias in multiplication */
  /* No rounding away test - it fails because of upward bias in multiplication */

  TEST(QS8_Q31__NEON, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_q31__neon);
  }

  TEST(QS8_Q31__NEON, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesApproximate(xnn_qs8_requantize_q31__neon);
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
   * Q31-based WAsm SIMD implementation.
   */

  TEST(QS8_Q31__WASMSIMD, exact_divide_by_po2) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .s(s)
        .TestExactDivideByPO2(xnn_qs8_requantize_q31__wasmsimd);
    }
  }

  TEST(QS8_Q31__WASMSIMD, exact_divide_by_po2_with_zero_point) {
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
          .TestExactDivideByPO2(xnn_qs8_requantize_q31__wasmsimd);
      }
    }
  }

  TEST(QS8_Q31__WASMSIMD, divide_by_po2_with_rounding_up) {
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
          .TestDivideByPO2WithRoundingUp(xnn_qs8_requantize_q31__wasmsimd);
      }
    }
  }

  /* No rounding down test - it fails because of upward bias in multiplication */
  /* No rounding away test - it fails because of upward bias in multiplication */

  TEST(QS8_Q31__WASMSIMD, special_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .TestSpecialCases(xnn_qs8_requantize_q31__wasmsimd);
  }

  TEST(QS8_Q31__WASMSIMD, random_cases) {
    RequantizationTester()
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(100)
      .TestRandomCasesApproximate(xnn_qs8_requantize_q31__wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD
