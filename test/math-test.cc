// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <climits>
#include <cstdint>

#include <gtest/gtest.h>
#include "src/xnnpack/math.h"
#include "src/xnnpack/requantization.h"
#include "test/next_prime.h"

namespace xnnpack {

TEST(math, rotl_u32) {
  EXPECT_EQ(math_rotl_u32(0x12345678u, 0), 0x12345678u);
  EXPECT_EQ(math_rotl_u32(0x12345678u, 4), 0x23456781u);
  EXPECT_EQ(math_rotl_u32(0x80000000u, 1), 0x00000001u);
  EXPECT_EQ(math_rotl_u32(0x00000001u, 31), 0x80000000u);
}

TEST(math, clz_u32) {
  EXPECT_EQ(math_clz_u32(0), 32);
  EXPECT_EQ(math_clz_u32(1), 31);
  EXPECT_EQ(math_clz_u32(0x80000000u), 0);
  EXPECT_EQ(math_clz_nonzero_u32(1), 31);
  EXPECT_EQ(math_clz_nonzero_u32(0x80000000u), 0);
}

TEST(math, ctz_u32) {
  EXPECT_EQ(math_ctz_u32(0), 32);
  EXPECT_EQ(math_ctz_u32(1), 0);
  EXPECT_EQ(math_ctz_u32(2), 1);
  EXPECT_EQ(math_ctz_u32(0x80000000u), 31);
  EXPECT_EQ(math_ctz_u32(0x00000080u), 7);
  EXPECT_EQ(math_ctz_nonzero_u32(1), 0);
  EXPECT_EQ(math_ctz_nonzero_u32(0x80000000u), 31);
}

TEST(math, abs_s32) {
  EXPECT_EQ(math_abs_s32(0), 0u);
  EXPECT_EQ(math_abs_s32(1), 1u);
  EXPECT_EQ(math_abs_s32(-1), 1u);
  EXPECT_EQ(math_abs_s32(INT32_MAX), static_cast<uint32_t>(INT32_MAX));
  EXPECT_EQ(math_abs_s32(INT32_MIN), 2147483648u);
}

TEST(math, asr_s32_rounding) {
  EXPECT_EQ(math_asr_s32_rounding(10, 0), 10);
  EXPECT_EQ(math_asr_s32_rounding(-10, 0), -10);
  EXPECT_EQ(math_asr_s32_rounding(10, 1), 5);
  EXPECT_EQ(math_asr_s32_rounding(11, 1), 6);
  EXPECT_EQ(math_asr_s32_rounding(-11, 1), -5);
  EXPECT_EQ(math_asr_s32_rounding(INT32_MAX, 32), 0);
  EXPECT_EQ(math_asr_s32_rounding(INT32_MIN, 32), 0);
  EXPECT_EQ(math_asr_s32_rounding(123456, 33), 0);
}

TEST(math, saturating_rounding_shift_left_s32) {
  EXPECT_EQ(saturating_rounding_shift_left_s32(-5, 4), -80);
  EXPECT_EQ(saturating_rounding_shift_left_s32(5, 4), 80);
  EXPECT_EQ(saturating_rounding_shift_left_s32(100, 32), INT32_MAX);
  EXPECT_EQ(saturating_rounding_shift_left_s32(-100, 32), INT32_MIN);
  EXPECT_EQ(saturating_rounding_shift_left_s32(0, 32), 0);
  EXPECT_EQ(saturating_rounding_shift_left_s32(10, -1), 5);
}

TEST(requantization, multiply_2x_high_s16) {
  // Edge case: -32768 * -32768 must saturate to 32767 (INT16_MAX),
  // matching x86 pmulhrsw and ARM NEON vqrdmulh.
  EXPECT_EQ(multiply_2x_high_s16(-32768, -32768), 32767);
  EXPECT_EQ(multiply_2x_high_s16(16384, 16384), 8192);
  EXPECT_EQ(multiply_2x_high_s16(-16384, 16384), -8192);
  EXPECT_EQ(multiply_2x_high_s16(0, -32768), 0);
}

TEST(next_prime, is_prime) {
  EXPECT_FALSE(IsPrime(0));
  EXPECT_FALSE(IsPrime(1));
  EXPECT_TRUE(IsPrime(2));
  EXPECT_TRUE(IsPrime(3));
  EXPECT_FALSE(IsPrime(4));
  EXPECT_TRUE(IsPrime(5));
  EXPECT_FALSE(IsPrime(9));
  EXPECT_TRUE(IsPrime(7919));
}

}  // namespace xnnpack
