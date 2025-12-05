// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neon.h"

#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_LOAD_STORE(arm_neon, u8x16, /*arch_flags=*/0);
TEST_LOAD_STORE(arm_neon, s8x16, /*arch_flags=*/0);
TEST_LOAD_STORE(arm_neon, s16x8, /*arch_flags=*/0);
TEST_LOAD_STORE(arm_neon, f16x8, /*arch_flags=*/0);
TEST_LOAD_STORE(arm_neon, bf16x8, /*arch_flags=*/0);
TEST_LOAD_STORE(arm_neon, f32x4, /*arch_flags=*/0);
TEST_LOAD_STORE(arm_neon, s32x4, /*arch_flags=*/0);

TEST_ALIGNED_LOAD_STORE(arm_neon, u8x16, /*arch_flags=*/0);
TEST_ALIGNED_LOAD_STORE(arm_neon, s8x16, /*arch_flags=*/0);
TEST_ALIGNED_LOAD_STORE(arm_neon, s16x8, /*arch_flags=*/0);
TEST_ALIGNED_LOAD_STORE(arm_neon, f16x8, /*arch_flags=*/0);
TEST_ALIGNED_LOAD_STORE(arm_neon, bf16x8, /*arch_flags=*/0);
TEST_ALIGNED_LOAD_STORE(arm_neon, f32x4, /*arch_flags=*/0);
TEST_ALIGNED_LOAD_STORE(arm_neon, s32x4, /*arch_flags=*/0);

TEST_PARTIAL_LOAD_STORE(arm_neon, u8x16, /*arch_flags=*/0);
TEST_PARTIAL_LOAD_STORE(arm_neon, s8x16, /*arch_flags=*/0);
TEST_PARTIAL_LOAD_STORE(arm_neon, s16x8, /*arch_flags=*/0);
TEST_PARTIAL_LOAD_STORE(arm_neon, f16x8, /*arch_flags=*/0);
TEST_PARTIAL_LOAD_STORE(arm_neon, bf16x8, /*arch_flags=*/0);
TEST_PARTIAL_LOAD_STORE(arm_neon, f32x4, /*arch_flags=*/0);
TEST_PARTIAL_LOAD_STORE(arm_neon, s32x4, /*arch_flags=*/0);

TEST_ADD(arm_neon, u8x16, /*arch_flags=*/0);
TEST_ADD(arm_neon, s8x16, /*arch_flags=*/0);
TEST_ADD(arm_neon, f32x4, /*arch_flags=*/0);
TEST_ADD(arm_neon, s32x4, /*arch_flags=*/0);

TEST_SUBTRACT(arm_neon, u8x16, /*arch_flags=*/0);
TEST_SUBTRACT(arm_neon, s8x16, /*arch_flags=*/0);
TEST_SUBTRACT(arm_neon, f32x4, /*arch_flags=*/0);
TEST_SUBTRACT(arm_neon, s32x4, /*arch_flags=*/0);

TEST_MULTIPLY(arm_neon, u8x16, /*arch_flags=*/0);
TEST_MULTIPLY(arm_neon, s8x16, /*arch_flags=*/0);
TEST_MULTIPLY(arm_neon, f32x4, /*arch_flags=*/0);
TEST_MULTIPLY(arm_neon, s32x4, /*arch_flags=*/0);

TEST_MIN(arm_neon, u8x16, /*arch_flags=*/0);
TEST_MIN(arm_neon, s8x16, /*arch_flags=*/0);
TEST_MIN(arm_neon, s16x8, /*arch_flags=*/0);
TEST_MIN(arm_neon, f32x4, /*arch_flags=*/0);
TEST_MIN(arm_neon, s32x4, /*arch_flags=*/0);

TEST_MAX(arm_neon, u8x16, /*arch_flags=*/0);
TEST_MAX(arm_neon, s8x16, /*arch_flags=*/0);
TEST_MAX(arm_neon, s16x8, /*arch_flags=*/0);
TEST_MAX(arm_neon, f32x4, /*arch_flags=*/0);
TEST_MAX(arm_neon, s32x4, /*arch_flags=*/0);

TEST_HORIZONTAL_MIN(arm_neon, u8x16, /*arch_flags=*/0);
TEST_HORIZONTAL_MIN(arm_neon, s8x16, /*arch_flags=*/0);
TEST_HORIZONTAL_MIN(arm_neon, s16x8, /*arch_flags=*/0);
TEST_HORIZONTAL_MIN(arm_neon, f32x4, /*arch_flags=*/0);
TEST_HORIZONTAL_MIN(arm_neon, s32x4, /*arch_flags=*/0);

TEST_HORIZONTAL_MAX(arm_neon, u8x16, /*arch_flags=*/0);
TEST_HORIZONTAL_MAX(arm_neon, s8x16, /*arch_flags=*/0);
TEST_HORIZONTAL_MAX(arm_neon, s16x8, /*arch_flags=*/0);
TEST_HORIZONTAL_MAX(arm_neon, f32x4, /*arch_flags=*/0);
TEST_HORIZONTAL_MAX(arm_neon, s32x4, /*arch_flags=*/0);

}  // namespace simd
}  // namespace ynn
