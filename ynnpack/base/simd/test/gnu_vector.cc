// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/gnu_vector.h"

#include <cmath>

#include <gtest/gtest.h>

// This must be included last
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class gnu_vector : public ::testing::Test {};

// Test 128-bit vector lengths (standard SIMD)
TEST_BROADCAST(gnu_vector, u8, 16);
TEST_BROADCAST(gnu_vector, s8, 16);
TEST_BROADCAST(gnu_vector, s16, 8);
TEST_BROADCAST(gnu_vector, u16, 8);
TEST_BROADCAST(gnu_vector, f32, 4);
TEST_BROADCAST(gnu_vector, s32, 4);
TEST_BROADCAST(gnu_vector, u32, 4);
TEST_BROADCAST(gnu_vector, f64, 2);

TEST_LOAD_STORE(gnu_vector, u8, 16);
TEST_LOAD_STORE(gnu_vector, s8, 16);
TEST_LOAD_STORE(gnu_vector, s16, 8);
TEST_LOAD_STORE(gnu_vector, u16, 8);
TEST_LOAD_STORE(gnu_vector, f32, 4);
TEST_LOAD_STORE(gnu_vector, s32, 4);
TEST_LOAD_STORE(gnu_vector, u32, 4);
TEST_LOAD_STORE(gnu_vector, f64, 2);

TEST_ALIGNED_LOAD_STORE(gnu_vector, u8, 16);
TEST_ALIGNED_LOAD_STORE(gnu_vector, s8, 16);
TEST_ALIGNED_LOAD_STORE(gnu_vector, s16, 8);
TEST_ALIGNED_LOAD_STORE(gnu_vector, u16, 8);
TEST_ALIGNED_LOAD_STORE(gnu_vector, f32, 4);
TEST_ALIGNED_LOAD_STORE(gnu_vector, s32, 4);
TEST_ALIGNED_LOAD_STORE(gnu_vector, u32, 4);
TEST_ALIGNED_LOAD_STORE(gnu_vector, f64, 2);

TEST_PARTIAL_LOAD_STORE(gnu_vector, u8, 16);
TEST_PARTIAL_LOAD_STORE(gnu_vector, s8, 16);
TEST_PARTIAL_LOAD_STORE(gnu_vector, s16, 8);
TEST_PARTIAL_LOAD_STORE(gnu_vector, u16, 8);
TEST_PARTIAL_LOAD_STORE(gnu_vector, f32, 4);
TEST_PARTIAL_LOAD_STORE(gnu_vector, s32, 4);
TEST_PARTIAL_LOAD_STORE(gnu_vector, u32, 4);
TEST_PARTIAL_LOAD_STORE(gnu_vector, f64, 2);

TEST_ADD(gnu_vector, u8, 16);
TEST_ADD(gnu_vector, s8, 16);
TEST_ADD(gnu_vector, s16, 8);
TEST_ADD(gnu_vector, u16, 8);
TEST_ADD(gnu_vector, f32, 4);
TEST_ADD(gnu_vector, s32, 4);
TEST_ADD(gnu_vector, u32, 4);
TEST_ADD(gnu_vector, f64, 2);

TEST_SUBTRACT(gnu_vector, u8, 16);
TEST_SUBTRACT(gnu_vector, s8, 16);
TEST_SUBTRACT(gnu_vector, s16, 8);
TEST_SUBTRACT(gnu_vector, u16, 8);
TEST_SUBTRACT(gnu_vector, f32, 4);
TEST_SUBTRACT(gnu_vector, s32, 4);
TEST_SUBTRACT(gnu_vector, u32, 4);
TEST_SUBTRACT(gnu_vector, f64, 2);

TEST_ADD_SAT(gnu_vector, u8, 16);
TEST_ADD_SAT(gnu_vector, s8, 16);
TEST_ADD_SAT(gnu_vector, u16, 8);
TEST_ADD_SAT(gnu_vector, s16, 8);

TEST_SUB_SAT(gnu_vector, u8, 16);
TEST_SUB_SAT(gnu_vector, s8, 16);
TEST_SUB_SAT(gnu_vector, u16, 8);
TEST_SUB_SAT(gnu_vector, s16, 8);

TEST_MULTIPLY(gnu_vector, u8, 16);
TEST_MULTIPLY(gnu_vector, s8, 16);
TEST_MULTIPLY(gnu_vector, s16, 8);
TEST_MULTIPLY(gnu_vector, u16, 8);
TEST_MULTIPLY(gnu_vector, f32, 4);
TEST_MULTIPLY(gnu_vector, s32, 4);
TEST_MULTIPLY(gnu_vector, u32, 4);
TEST_MULTIPLY(gnu_vector, f64, 2);

TEST_DIVIDE(gnu_vector, f32, 4);
TEST_DIVIDE(gnu_vector, f64, 2);

TEST_COPYSIGN(gnu_vector, f32, 4);
TEST_COPYSIGN(gnu_vector, f64, 2);

TEST_MIN(gnu_vector, u8, 16);
TEST_MIN(gnu_vector, s8, 16);
TEST_MIN(gnu_vector, u16, 8);
TEST_MIN(gnu_vector, s16, 8);
TEST_MIN(gnu_vector, u32, 4);
TEST_MIN(gnu_vector, s32, 4);
TEST_MIN(gnu_vector, f32, 4);
TEST_MIN(gnu_vector, f64, 2);

TEST_MAX(gnu_vector, u8, 16);
TEST_MAX(gnu_vector, s8, 16);
TEST_MAX(gnu_vector, u16, 8);
TEST_MAX(gnu_vector, s16, 8);
TEST_MAX(gnu_vector, u32, 4);
TEST_MAX(gnu_vector, s32, 4);
TEST_MAX(gnu_vector, f32, 4);
TEST_MAX(gnu_vector, f64, 2);

TEST_AND(gnu_vector, u8, 16);
TEST_AND(gnu_vector, s8, 16);
TEST_AND(gnu_vector, s16, 8);
TEST_AND(gnu_vector, u16, 8);
TEST_AND(gnu_vector, s32, 4);
TEST_AND(gnu_vector, u32, 4);

TEST_OR(gnu_vector, u8, 16);
TEST_OR(gnu_vector, s8, 16);
TEST_OR(gnu_vector, s16, 8);
TEST_OR(gnu_vector, u16, 8);
TEST_OR(gnu_vector, s32, 4);
TEST_OR(gnu_vector, u32, 4);

TEST_XOR(gnu_vector, u8, 16);
TEST_XOR(gnu_vector, s8, 16);
TEST_XOR(gnu_vector, s16, 8);
TEST_XOR(gnu_vector, u16, 8);
TEST_XOR(gnu_vector, s32, 4);
TEST_XOR(gnu_vector, u32, 4);

TEST_NOT(gnu_vector, u8, 16);
TEST_NOT(gnu_vector, s8, 16);
TEST_NOT(gnu_vector, s16, 8);
TEST_NOT(gnu_vector, u16, 8);
TEST_NOT(gnu_vector, s32, 4);
TEST_NOT(gnu_vector, u32, 4);

TEST_SHIFT_LEFT(gnu_vector, s8, 16);
TEST_SHIFT_LEFT(gnu_vector, u8, 16);
TEST_SHIFT_LEFT(gnu_vector, s16, 8);
TEST_SHIFT_LEFT(gnu_vector, u16, 8);
TEST_SHIFT_LEFT(gnu_vector, s32, 4);
TEST_SHIFT_LEFT(gnu_vector, u32, 4);

TEST_SQRT(gnu_vector, f32, 4);
TEST_SQRT(gnu_vector, f64, 2);

TEST_ABS(gnu_vector, s8, 16);
TEST_ABS(gnu_vector, s16, 8);
TEST_ABS(gnu_vector, s32, 4);
TEST_ABS(gnu_vector, f32, 4);
TEST_ABS(gnu_vector, f64, 2);

TEST_FLOOR_LOG2(gnu_vector, f32, 4);
TEST_FLOOR_LOG2(gnu_vector, f64, 2);

TEST_EXP2_ROUND(gnu_vector, f32, 4);
TEST_EXP2_ROUND(gnu_vector, f64, 2);

TEST_COMPARISONS(gnu_vector, u8, 16);
TEST_COMPARISONS(gnu_vector, s8, 16);
TEST_COMPARISONS(gnu_vector, s16, 8);
TEST_COMPARISONS(gnu_vector, u16, 8);
TEST_COMPARISONS(gnu_vector, s32, 4);
TEST_COMPARISONS(gnu_vector, u32, 4);
TEST_COMPARISONS(gnu_vector, f32, 4);
TEST_COMPARISONS(gnu_vector, f64, 2);

TEST_ISNAN(gnu_vector, f32, 4);
TEST_ISNAN(gnu_vector, f64, 2);

TEST_ISFINITE(gnu_vector, f32, 4);
TEST_ISFINITE(gnu_vector, f64, 2);

TEST_HORIZONTAL_MIN(gnu_vector, u8, 16);
TEST_HORIZONTAL_MIN(gnu_vector, s8, 16);
TEST_HORIZONTAL_MIN(gnu_vector, u16, 8);
TEST_HORIZONTAL_MIN(gnu_vector, s16, 8);
TEST_HORIZONTAL_MIN(gnu_vector, u32, 4);
TEST_HORIZONTAL_MIN(gnu_vector, s32, 4);
TEST_HORIZONTAL_MIN(gnu_vector, f32, 4);
TEST_HORIZONTAL_MIN(gnu_vector, f64, 2);

TEST_HORIZONTAL_MAX(gnu_vector, u8, 16);
TEST_HORIZONTAL_MAX(gnu_vector, s8, 16);
TEST_HORIZONTAL_MAX(gnu_vector, u16, 8);
TEST_HORIZONTAL_MAX(gnu_vector, s16, 8);
TEST_HORIZONTAL_MAX(gnu_vector, u32, 4);
TEST_HORIZONTAL_MAX(gnu_vector, s32, 4);
TEST_HORIZONTAL_MAX(gnu_vector, f32, 4);
TEST_HORIZONTAL_MAX(gnu_vector, f64, 2);

TEST_HORIZONTAL_SUM(gnu_vector, u8, 16);
TEST_HORIZONTAL_SUM(gnu_vector, s8, 16);
TEST_HORIZONTAL_SUM(gnu_vector, u16, 8);
TEST_HORIZONTAL_SUM(gnu_vector, s16, 8);
TEST_HORIZONTAL_SUM(gnu_vector, s32, 4);
TEST_HORIZONTAL_SUM(gnu_vector, f32, 4);
TEST_HORIZONTAL_SUM(gnu_vector, f64, 2);

TEST_UNARY(gnu_vector, exp, f32, 4, std::exp, 2);
TEST_UNARY(gnu_vector, exp, f64, 2, std::exp, 2);
TEST_UNARY(gnu_vector, expm1, f32, 4, std::expm1, 2);
TEST_UNARY(gnu_vector, expm1, f64, 2, std::expm1, 2);
TEST_UNARY(gnu_vector, log, f32, 4, std::log, 2);
TEST_UNARY(gnu_vector, log, f64, 2, std::log, 2);
TEST_UNARY(gnu_vector, log1p, f32, 4, std::log1p, 3);
TEST_UNARY(gnu_vector, log1p, f64, 2, std::log1p, 3);
TEST_UNARY(gnu_vector, erf, f32, 4, std::erf, 2);
TEST_UNARY(gnu_vector, erf, f64, 2, std::erf, 3);
TEST_UNARY(gnu_vector, tanh, f32, 4, std::tanh, 3);
TEST_UNARY(gnu_vector, tanh, f64, 2, std::tanh, 4);

TEST_UNARY(gnu_vector, approx_erf, f32, 4, std::erf, 5);
TEST_UNARY(gnu_vector, approx_tanh, f32, 4, std::tanh, 5);

}  // namespace simd
}  // namespace ynn
