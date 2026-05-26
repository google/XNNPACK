// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/arm_vec128.h"

// This must be included last
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class arm_neon : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::neon)) {
      GTEST_SKIP() << "neon not supported on this hardware";
    }
  }
};

TEST_BROADCAST(arm_neon, u8, 16);
TEST_BROADCAST(arm_neon, s8, 16);
TEST_BROADCAST(arm_neon, s16, 8);
TEST_BROADCAST(arm_neon, f16, 8);
TEST_BROADCAST(arm_neon, bf16, 8);
TEST_BROADCAST(arm_neon, f32, 4);
TEST_BROADCAST(arm_neon, s32, 4);

TEST_BROADCAST(arm_neon, u8, 8);

TEST_LOAD_STORE(arm_neon, u8, 16);
TEST_LOAD_STORE(arm_neon, s8, 16);
TEST_LOAD_STORE(arm_neon, s16, 8);
TEST_LOAD_STORE(arm_neon, f16, 8);
TEST_LOAD_STORE(arm_neon, bf16, 8);
TEST_LOAD_STORE(arm_neon, f32, 4);
TEST_LOAD_STORE(arm_neon, s32, 4);

TEST_LOAD_STORE(arm_neon, u8, 8);

TEST_ALIGNED_LOAD_STORE(arm_neon, u8, 16);
TEST_ALIGNED_LOAD_STORE(arm_neon, s8, 16);
TEST_ALIGNED_LOAD_STORE(arm_neon, s16, 8);
TEST_ALIGNED_LOAD_STORE(arm_neon, f16, 8);
TEST_ALIGNED_LOAD_STORE(arm_neon, bf16, 8);
TEST_ALIGNED_LOAD_STORE(arm_neon, f32, 4);
TEST_ALIGNED_LOAD_STORE(arm_neon, s32, 4);

TEST_ALIGNED_LOAD_STORE(arm_neon, u8, 8);

TEST_PARTIAL_LOAD_STORE(arm_neon, u8, 16);
TEST_PARTIAL_LOAD_STORE(arm_neon, s8, 16);
TEST_PARTIAL_LOAD_STORE(arm_neon, s16, 8);
TEST_PARTIAL_LOAD_STORE(arm_neon, f16, 8);
TEST_PARTIAL_LOAD_STORE(arm_neon, bf16, 8);
TEST_PARTIAL_LOAD_STORE(arm_neon, f32, 4);
TEST_PARTIAL_LOAD_STORE(arm_neon, s32, 4);

TEST_PARTIAL_LOAD_STORE(arm_neon, u8, 8);

TEST_ADD(arm_neon, u8, 16);
TEST_ADD(arm_neon, s8, 16);
TEST_ADD(arm_neon, f32, 4);
TEST_ADD(arm_neon, s32, 4);

TEST_SUBTRACT(arm_neon, u8, 16);
TEST_SUBTRACT(arm_neon, s8, 16);
TEST_SUBTRACT(arm_neon, f32, 4);
TEST_SUBTRACT(arm_neon, s32, 4);

TEST_ADD_SAT(arm_neon, u8, 16);
TEST_ADD_SAT(arm_neon, s8, 16);
TEST_ADD_SAT(arm_neon, u16, 8);
TEST_ADD_SAT(arm_neon, s16, 8);
TEST_ADD_SAT(arm_neon, u32, 4);
TEST_ADD_SAT(arm_neon, s32, 4);

TEST_SUB_SAT(arm_neon, u8, 16);
TEST_SUB_SAT(arm_neon, s8, 16);
TEST_SUB_SAT(arm_neon, u16, 8);
TEST_SUB_SAT(arm_neon, s16, 8);
TEST_SUB_SAT(arm_neon, u32, 4);
TEST_SUB_SAT(arm_neon, s32, 4);

TEST_MULTIPLY(arm_neon, u8, 16);
TEST_MULTIPLY(arm_neon, s8, 16);
TEST_MULTIPLY(arm_neon, f32, 4);
TEST_MULTIPLY(arm_neon, s32, 4);

TEST_DIVIDE(arm_neon, f32, 4);

TEST_MIN(arm_neon, u8, 16);
TEST_MIN(arm_neon, s8, 16);
TEST_MIN(arm_neon, s16, 8);
TEST_MIN(arm_neon, f32, 4);
TEST_MIN(arm_neon, s32, 4);

TEST_MAX(arm_neon, u8, 16);
TEST_MAX(arm_neon, s8, 16);
TEST_MAX(arm_neon, s16, 8);
TEST_MAX(arm_neon, f32, 4);
TEST_MAX(arm_neon, s32, 4);

TEST_AND(arm_neon, u8, 16);
TEST_AND(arm_neon, s8, 16);
TEST_AND(arm_neon, s16, 8);
TEST_AND(arm_neon, s32, 4);
TEST_AND(arm_neon, u8, 8);

TEST_OR(arm_neon, u8, 16);
TEST_OR(arm_neon, s8, 16);
TEST_OR(arm_neon, s16, 8);
TEST_OR(arm_neon, s32, 4);
TEST_OR(arm_neon, u8, 8);

TEST_XOR(arm_neon, u8, 16);
TEST_XOR(arm_neon, s8, 16);
TEST_XOR(arm_neon, s16, 8);
TEST_XOR(arm_neon, s32, 4);
TEST_XOR(arm_neon, u8, 8);

TEST_NOT(arm_neon, u8, 16);
TEST_NOT(arm_neon, s8, 16);
TEST_NOT(arm_neon, s16, 8);
TEST_NOT(arm_neon, s32, 4);
TEST_NOT(arm_neon, u8, 8);

TEST_SHIFT_LEFT(arm_neon, s8, 16);
TEST_SHIFT_LEFT(arm_neon, s16, 8);
TEST_SHIFT_LEFT(arm_neon, s32, 4);

TEST_FLOOR(arm_neon, f32, 4);
TEST_CEIL(arm_neon, f32, 4);
TEST_ROUND(arm_neon, f32, 4);
TEST_SQRT(arm_neon, f32, 4);

TEST_ABS(arm_neon, s8, 16);
TEST_ABS(arm_neon, s16, 8);
TEST_ABS(arm_neon, s32, 4);
TEST_ABS(arm_neon, f32, 4);

TEST_COPYSIGN(arm_neon, f32, 4);

TEST_FLOOR_LOG2(arm_neon, f32, 4);
TEST_EXP2_ROUND(arm_neon, f32, 4);
TEST_COMPARISONS(arm_neon, f32, 4);
TEST_ISNAN(arm_neon, f32, 4);
TEST_ISFINITE(arm_neon, f32, 4);

TEST_HORIZONTAL_MIN(arm_neon, u8, 16);
TEST_HORIZONTAL_MIN(arm_neon, s8, 16);
TEST_HORIZONTAL_MIN(arm_neon, s16, 8);
TEST_HORIZONTAL_MIN(arm_neon, f32, 4);
TEST_HORIZONTAL_MIN(arm_neon, s32, 4);

TEST_HORIZONTAL_MAX(arm_neon, u8, 16);
TEST_HORIZONTAL_MAX(arm_neon, s8, 16);
TEST_HORIZONTAL_MAX(arm_neon, s16, 8);
TEST_HORIZONTAL_MAX(arm_neon, f32, 4);
TEST_HORIZONTAL_MAX(arm_neon, s32, 4);

TEST_CAST(arm_neon, s32, s8x16);
TEST_CAST(arm_neon, s32, u8x16);
TEST_CAST(arm_neon, s32, s16x8);
TEST_CAST(arm_neon, f32, s32x4);
TEST_CAST(arm_neon, s32, f32x4);
TEST_CAST(arm_neon, f32, bf16x8);
TEST_CAST(arm_neon, bf16, f32x8);

TEST_CAST(arm_neon, s16, s32x8);
TEST_CAST(arm_neon, u8, s16x16);
TEST_CAST(arm_neon, s8, s16x16);
TEST_CAST(arm_neon, u8, f32x16);
TEST_CAST(arm_neon, s8, f32x16);
TEST_CAST(arm_neon, s16, f32x8);

TEST_EXTRACT(arm_neon, u8x16, 8);

TEST_CONCAT(arm_neon, u8x8);

#ifdef YNN_ARCH_ARM64
TEST_UNARY(arm_neon, exp, f32, 4, std::exp, 2);
#else
// TODO: b/515053903 - 32-bit ARM does something weird here.
#endif
TEST_UNARY(arm_neon, expm1, f32, 4, std::expm1, 2);
TEST_UNARY(arm_neon, log, f32, 4, std::log, 2);
TEST_UNARY(arm_neon, log1p, f32, 4, std::log1p, 3);
TEST_UNARY(arm_neon, erf, f32, 4, std::erf, 2);
TEST_UNARY(arm_neon, tanh, f32, 4, std::tanh, 5);

TEST_UNARY(arm_neon, fast_erf, f32, 4, std::erf, 5);

}  // namespace simd
}  // namespace ynn
