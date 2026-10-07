// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "src/xnnpack/reference-utils.h"

namespace xnnpack {

TEST(ReferenceUtils, EuclideanDivInt32MinByMinusOne) {
  // INT32_MIN / -1 must not cause CPU division overflow (SIGFPE).
  EXPECT_EQ(euclidean_div<int32_t>(std::numeric_limits<int32_t>::min(), -1),
            std::numeric_limits<int32_t>::min());
}

TEST(ReferenceUtils, EuclideanDivByZero) {
  EXPECT_EQ(euclidean_div<int32_t>(10, 0), 0);
  EXPECT_EQ(euclidean_div<int32_t>(-10, 0), 0);
  EXPECT_EQ(euclidean_div<int32_t>(0, 0), 0);
}

TEST(ReferenceUtils, EuclideanDivGeneral) {
  EXPECT_EQ(euclidean_div<int32_t>(7, 3), 2);
  EXPECT_EQ(euclidean_div<int32_t>(-7, 3), -3);
  EXPECT_EQ(euclidean_div<int32_t>(7, -3), -2);
  EXPECT_EQ(euclidean_div<int32_t>(-7, -3), 3);
}

TEST(ReferenceUtils, EuclideanModInt32MinByMinusOne) {
  // INT32_MIN % -1 must not cause CPU division overflow (SIGFPE).
  EXPECT_EQ(euclidean_mod<int32_t>(std::numeric_limits<int32_t>::min(), -1),
            0);
}

TEST(ReferenceUtils, EuclideanModByZeroOrOneOrMinusOne) {
  EXPECT_EQ(euclidean_mod<int32_t>(42, 0), 0);
  EXPECT_EQ(euclidean_mod<int32_t>(42, 1), 0);
  EXPECT_EQ(euclidean_mod<int32_t>(42, -1), 0);
  EXPECT_EQ(euclidean_mod<int32_t>(-42, 1), 0);
  EXPECT_EQ(euclidean_mod<int32_t>(-42, -1), 0);
}

TEST(ReferenceUtils, EuclideanModGeneral) {
  EXPECT_EQ(euclidean_mod<int32_t>(7, 3), 1);
  EXPECT_EQ(euclidean_mod<int32_t>(-7, 3), 2);
  EXPECT_EQ(euclidean_mod<int32_t>(7, -3), 1);
  EXPECT_EQ(euclidean_mod<int32_t>(-7, -3), 2);
}

TEST(ReferenceUtils, IntegerPowMinExponent) {
  // Minimum negative exponent must not overflow -INT32_MIN.
  EXPECT_EQ(integer_pow<int32_t>(1, std::numeric_limits<int32_t>::min()), 1);
  EXPECT_EQ(integer_pow<int32_t>(-1, std::numeric_limits<int32_t>::min()), 1);
  EXPECT_EQ(integer_pow<int32_t>(2, std::numeric_limits<int32_t>::min()), 0);
}

TEST(ReferenceUtils, IntegerPowNegativeExponent) {
  EXPECT_EQ(integer_pow<int32_t>(1, -5), 1);
  EXPECT_EQ(integer_pow<int32_t>(-1, -5), -1);
  EXPECT_EQ(integer_pow<int32_t>(-1, -4), 1);
  EXPECT_EQ(integer_pow<int32_t>(2, -3), 0);
}

TEST(ReferenceUtils, IntegerPowPositive) {
  EXPECT_EQ(integer_pow<int32_t>(2, 0), 1);
  EXPECT_EQ(integer_pow<int32_t>(2, 1), 2);
  EXPECT_EQ(integer_pow<int32_t>(2, 10), 1024);
  EXPECT_EQ(integer_pow<int32_t>(-3, 3), -27);
  EXPECT_EQ(integer_pow<int32_t>(-3, 4), 81);
}

TEST(ReferenceUtils, IntegerPowModularWrapNoUBSan) {
  // Squaring INT32_MIN using unsigned arithmetic wraps cleanly without UBSan.
  EXPECT_EQ(integer_pow<int32_t>(std::numeric_limits<int32_t>::min(), 2), 0);
}

}  // namespace xnnpack
