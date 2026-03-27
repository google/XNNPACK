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
TEST_LOAD_STORE(wasm_simd128, s32, 4);
TEST_LOAD_STORE(wasm_simd128, f64, 2);

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
TEST_MIN(wasm_simd128, s16, 8);
TEST_MIN(wasm_simd128, u8, 16);

TEST_MAX(wasm_simd128, f32, 4);
TEST_MAX(wasm_simd128, s16, 8);
TEST_MAX(wasm_simd128, u8, 16);

}  // namespace simd
}  // namespace ynn
