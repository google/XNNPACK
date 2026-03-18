// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse2.h"

#include <gtest/gtest.h>
#include "ynnpack/base/simd/test/generic.h"
#include "ynnpack/base/simd/x86_sse2_fma.h"

namespace ynn {
namespace simd {

// We assume we can always use sse2.
class x86_sse2 : public ::testing::Test {};

TEST_BROADCAST(x86_sse2, u8, 16);
TEST_BROADCAST(x86_sse2, s8, 16);
TEST_BROADCAST(x86_sse2, s16, 8);
TEST_BROADCAST(x86_sse2, f16, 8);
TEST_BROADCAST(x86_sse2, bf16, 8);
TEST_BROADCAST(x86_sse2, f32, 4);
TEST_BROADCAST(x86_sse2, s32, 4);
TEST_BROADCAST(x86_sse2, f64, 2);

TEST_LOAD_STORE(x86_sse2, u8, 16);
TEST_LOAD_STORE(x86_sse2, s8, 16);
TEST_LOAD_STORE(x86_sse2, s16, 8);
TEST_LOAD_STORE(x86_sse2, f16, 8);
TEST_LOAD_STORE(x86_sse2, bf16, 8);
TEST_LOAD_STORE(x86_sse2, f32, 4);
TEST_LOAD_STORE(x86_sse2, s32, 4);

TEST_ALIGNED_LOAD_STORE(x86_sse2, u8, 16);
TEST_ALIGNED_LOAD_STORE(x86_sse2, s8, 16);
TEST_ALIGNED_LOAD_STORE(x86_sse2, s16, 8);
TEST_ALIGNED_LOAD_STORE(x86_sse2, f16, 8);
TEST_ALIGNED_LOAD_STORE(x86_sse2, bf16, 8);
TEST_ALIGNED_LOAD_STORE(x86_sse2, f32, 4);
TEST_ALIGNED_LOAD_STORE(x86_sse2, s32, 4);

TEST_PARTIAL_LOAD_STORE(x86_sse2, u8, 16);
TEST_PARTIAL_LOAD_STORE(x86_sse2, s8, 16);
TEST_PARTIAL_LOAD_STORE(x86_sse2, s16, 8);
TEST_PARTIAL_LOAD_STORE(x86_sse2, f16, 8);
TEST_PARTIAL_LOAD_STORE(x86_sse2, bf16, 8);
TEST_PARTIAL_LOAD_STORE(x86_sse2, f32, 4);
TEST_PARTIAL_LOAD_STORE(x86_sse2, s32, 4);

TEST_ADD(x86_sse2, u8, 16);
TEST_ADD(x86_sse2, s8, 16);
TEST_ADD(x86_sse2, s16, 8);
TEST_ADD(x86_sse2, f32, 4);
TEST_ADD(x86_sse2, s32, 4);
TEST_ADD(x86_sse2, f64, 2);

TEST_SUBTRACT(x86_sse2, u8, 16);
TEST_SUBTRACT(x86_sse2, s8, 16);
TEST_SUBTRACT(x86_sse2, s16, 8);
TEST_SUBTRACT(x86_sse2, f32, 4);
TEST_SUBTRACT(x86_sse2, s32, 4);
TEST_SUBTRACT(x86_sse2, f64, 2);

TEST_MULTIPLY(x86_sse2, f32, 4);
TEST_MULTIPLY(x86_sse2, f64, 2);

TEST_COPYSIGN(x86_sse2, f64, 2);

TEST_MIN(x86_sse2, u8, 16);
TEST_MIN(x86_sse2, s16, 8);
TEST_MIN(x86_sse2, f32, 4);

TEST_MAX(x86_sse2, u8, 16);
TEST_MAX(x86_sse2, s16, 8);
TEST_MAX(x86_sse2, f32, 4);

TEST_AND(x86_sse2, u8, 16);
TEST_AND(x86_sse2, s8, 16);
TEST_AND(x86_sse2, s16, 8);
TEST_AND(x86_sse2, s32, 4);

TEST_OR(x86_sse2, u8, 16);
TEST_OR(x86_sse2, s8, 16);
TEST_OR(x86_sse2, s16, 8);
TEST_OR(x86_sse2, s32, 4);

TEST_XOR(x86_sse2, u8, 16);
TEST_XOR(x86_sse2, s8, 16);
TEST_XOR(x86_sse2, s16, 8);
TEST_XOR(x86_sse2, s32, 4);

TEST_NOT(x86_sse2, u8, 16);
TEST_NOT(x86_sse2, s8, 16);
TEST_NOT(x86_sse2, s16, 8);
TEST_NOT(x86_sse2, s32, 4);

TEST_SQRT(x86_sse2, f32, 4);

TEST_ABS(x86_sse2, f32, 4);

TEST_HORIZONTAL_MIN(x86_sse2, u8, 16);
TEST_HORIZONTAL_MIN(x86_sse2, s16, 8);
TEST_HORIZONTAL_MIN(x86_sse2, f32, 4);

TEST_HORIZONTAL_MAX(x86_sse2, u8, 16);
TEST_HORIZONTAL_MAX(x86_sse2, s16, 8);
TEST_HORIZONTAL_MAX(x86_sse2, f32, 4);

TEST_CONVERT(x86_sse2, f32, bf16x8);
TEST_CONVERT(x86_sse2, s32, u8x16);
TEST_CONVERT(x86_sse2, s32, s8x16);
TEST_CONVERT(x86_sse2, f32, s32x4);
TEST_CONVERT(x86_sse2, s32, f32x4);
TEST_CONVERT(x86_sse2, f64, f32x4);
TEST_CONVERT(x86_sse2, f32, f64x4);

TEST_FMA(x86_sse2, f32, 4);

}  // namespace simd
}  // namespace ynn
