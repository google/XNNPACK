// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/wasm_simd128.h"

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class wasm_simd128 : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::wasm_simd128)) {
      GTEST_SKIP() << "wasm simd128 not supported on this hardware";
    }
  }
};

TEST_BROADCAST(wasm_simd128, u8, 16);
TEST_BROADCAST(wasm_simd128, s8, 16);
TEST_BROADCAST(wasm_simd128, u16, 8);
TEST_BROADCAST(wasm_simd128, s16, 8);
TEST_BROADCAST(wasm_simd128, f32, 4);
TEST_BROADCAST(wasm_simd128, s32, 4);
TEST_BROADCAST(wasm_simd128, f64, 2);

TEST_LOAD_STORE(wasm_simd128, u8, 16);
TEST_LOAD_STORE(wasm_simd128, s8, 16);
TEST_LOAD_STORE(wasm_simd128, u16, 8);
TEST_LOAD_STORE(wasm_simd128, s16, 8);
TEST_LOAD_STORE(wasm_simd128, f32, 4);
TEST_LOAD_STORE(wasm_simd128, u32, 4);
TEST_LOAD_STORE(wasm_simd128, s32, 4);
TEST_LOAD_STORE(wasm_simd128, f64, 2);

TEST_ALIGNED_LOAD_STORE(wasm_simd128, u8, 16);
TEST_ALIGNED_LOAD_STORE(wasm_simd128, s8, 16);
TEST_ALIGNED_LOAD_STORE(wasm_simd128, u16, 8);
TEST_ALIGNED_LOAD_STORE(wasm_simd128, s16, 8);
TEST_ALIGNED_LOAD_STORE(wasm_simd128, f32, 4);
TEST_ALIGNED_LOAD_STORE(wasm_simd128, u32, 4);
TEST_ALIGNED_LOAD_STORE(wasm_simd128, s32, 4);
TEST_ALIGNED_LOAD_STORE(wasm_simd128, f64, 2);

TEST_PARTIAL_LOAD_STORE(wasm_simd128, u8, 16);
TEST_PARTIAL_LOAD_STORE(wasm_simd128, s8, 16);
TEST_PARTIAL_LOAD_STORE(wasm_simd128, s16, 8);
TEST_PARTIAL_LOAD_STORE(wasm_simd128, f32, 4);
TEST_PARTIAL_LOAD_STORE(wasm_simd128, s32, 4);

TEST_ADD(wasm_simd128, u8, 16);
TEST_ADD(wasm_simd128, s8, 16);
TEST_ADD(wasm_simd128, s16, 8);
TEST_ADD(wasm_simd128, f32, 4);
TEST_ADD(wasm_simd128, s32, 4);
TEST_ADD(wasm_simd128, f64, 2);

TEST_SUBTRACT(wasm_simd128, u8, 16);
TEST_SUBTRACT(wasm_simd128, s8, 16);
TEST_SUBTRACT(wasm_simd128, s16, 8);
TEST_SUBTRACT(wasm_simd128, f32, 4);
TEST_SUBTRACT(wasm_simd128, s32, 4);
TEST_SUBTRACT(wasm_simd128, f64, 2);

TEST_ADD_SAT(wasm_simd128, u8, 16);
TEST_ADD_SAT(wasm_simd128, s8, 16);
TEST_ADD_SAT(wasm_simd128, u16, 8);
TEST_ADD_SAT(wasm_simd128, s16, 8);

TEST_SUB_SAT(wasm_simd128, u8, 16);
TEST_SUB_SAT(wasm_simd128, s8, 16);
TEST_SUB_SAT(wasm_simd128, u16, 8);
TEST_SUB_SAT(wasm_simd128, s16, 8);

TEST_MULTIPLY(wasm_simd128, f32, 4);
TEST_MULTIPLY(wasm_simd128, f64, 2);

TEST_DIVIDE(wasm_simd128, f32, 4);

TEST_AND(wasm_simd128, u8, 16);
TEST_AND(wasm_simd128, s8, 16);
TEST_AND(wasm_simd128, s16, 8);
TEST_AND(wasm_simd128, s32, 4);

TEST_OR(wasm_simd128, u8, 16);
TEST_OR(wasm_simd128, s8, 16);
TEST_OR(wasm_simd128, s16, 8);
TEST_OR(wasm_simd128, s32, 4);

TEST_XOR(wasm_simd128, u8, 16);
TEST_XOR(wasm_simd128, s8, 16);
TEST_XOR(wasm_simd128, s16, 8);
TEST_XOR(wasm_simd128, s32, 4);

TEST_NOT(wasm_simd128, u8, 16);
TEST_NOT(wasm_simd128, s8, 16);
TEST_NOT(wasm_simd128, s16, 8);
TEST_NOT(wasm_simd128, s32, 4);

TEST_SHIFT_LEFT(wasm_simd128, s16, 8);
TEST_SHIFT_LEFT(wasm_simd128, s32, 4);

TEST_MIN(wasm_simd128, f32, 4);
TEST_MIN(wasm_simd128, u16, 8);
TEST_MIN(wasm_simd128, s16, 8);
TEST_MIN(wasm_simd128, u8, 16);
TEST_MIN(wasm_simd128, s8, 16);

TEST_MAX(wasm_simd128, f32, 4);
TEST_MAX(wasm_simd128, u16, 8);
TEST_MAX(wasm_simd128, s16, 8);
TEST_MAX(wasm_simd128, u8, 16);
TEST_MAX(wasm_simd128, s8, 16);

TEST_FLOOR(wasm_simd128, f32, 4);
TEST_CEIL(wasm_simd128, f32, 4);
TEST_ROUND(wasm_simd128, f32, 4);
TEST_SQRT(wasm_simd128, f32, 4);

TEST_ABS(wasm_simd128, s8, 16);
TEST_ABS(wasm_simd128, s16, 8);
TEST_ABS(wasm_simd128, s32, 4);
TEST_ABS(wasm_simd128, f32, 4);

TEST_HORIZONTAL_MIN(wasm_simd128, u8, 16);
TEST_HORIZONTAL_MIN(wasm_simd128, s8, 16);
TEST_HORIZONTAL_MIN(wasm_simd128, s16, 8);
TEST_HORIZONTAL_MIN(wasm_simd128, f32, 4);
TEST_HORIZONTAL_MIN(wasm_simd128, s32, 4);

TEST_HORIZONTAL_MAX(wasm_simd128, u8, 16);
TEST_HORIZONTAL_MAX(wasm_simd128, s8, 16);
TEST_HORIZONTAL_MAX(wasm_simd128, s16, 8);
TEST_HORIZONTAL_MAX(wasm_simd128, f32, 4);
TEST_HORIZONTAL_MAX(wasm_simd128, s32, 4);

TEST_CAST(wasm_simd128, s32, f32x4);
TEST_CAST(wasm_simd128, f32, s32x4);
TEST_CAST(wasm_simd128, s32, s8x16);
TEST_CAST(wasm_simd128, s32, u8x16);
TEST_CAST(wasm_simd128, s16, s8x16);
TEST_CAST(wasm_simd128, u16, u8x16);
TEST_CAST(wasm_simd128, s32, s16x8);
TEST_CAST(wasm_simd128, s32, u16x8);

TEST_SATURATE_CAST(wasm_simd128, s16, s32x8);
TEST_SATURATE_CAST(wasm_simd128, s8, s16x16);
TEST_SATURATE_CAST(wasm_simd128, u8, s16x16);

TEST_ROUND_FLOAT_TO_INT(wasm_simd128, s16, f32x8);
TEST_ROUND_FLOAT_TO_INT(wasm_simd128, s8, f32x16);
TEST_ROUND_FLOAT_TO_INT(wasm_simd128, u8, f32x16);

}  // namespace simd
}  // namespace ynn
