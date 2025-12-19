// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neon.h"

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_BROADCAST(arm_neon, uint8_t, 16);
TEST_BROADCAST(arm_neon, int8_t, 16);
TEST_BROADCAST(arm_neon, int16_t, 8);
TEST_BROADCAST(arm_neon, half, 8);
TEST_BROADCAST(arm_neon, bfloat16, 8);
TEST_BROADCAST(arm_neon, float, 4);
TEST_BROADCAST(arm_neon, int32_t, 4);

TEST_LOAD_STORE(arm_neon, uint8_t, 16);
TEST_LOAD_STORE(arm_neon, int8_t, 16);
TEST_LOAD_STORE(arm_neon, int16_t, 8);
TEST_LOAD_STORE(arm_neon, half, 8);
TEST_LOAD_STORE(arm_neon, bfloat16, 8);
TEST_LOAD_STORE(arm_neon, float, 4);
TEST_LOAD_STORE(arm_neon, int32_t, 4);

TEST_ALIGNED_LOAD_STORE(arm_neon, uint8_t, 16);
TEST_ALIGNED_LOAD_STORE(arm_neon, int8_t, 16);
TEST_ALIGNED_LOAD_STORE(arm_neon, int16_t, 8);
TEST_ALIGNED_LOAD_STORE(arm_neon, half, 8);
TEST_ALIGNED_LOAD_STORE(arm_neon, bfloat16, 8);
TEST_ALIGNED_LOAD_STORE(arm_neon, float, 4);
TEST_ALIGNED_LOAD_STORE(arm_neon, int32_t, 4);

TEST_PARTIAL_LOAD_STORE(arm_neon, uint8_t, 16);
TEST_PARTIAL_LOAD_STORE(arm_neon, int8_t, 16);
TEST_PARTIAL_LOAD_STORE(arm_neon, int16_t, 8);
TEST_PARTIAL_LOAD_STORE(arm_neon, half, 8);
TEST_PARTIAL_LOAD_STORE(arm_neon, bfloat16, 8);
TEST_PARTIAL_LOAD_STORE(arm_neon, float, 4);
TEST_PARTIAL_LOAD_STORE(arm_neon, int32_t, 4);

TEST_ADD(arm_neon, uint8_t, 16);
TEST_ADD(arm_neon, int8_t, 16);
TEST_ADD(arm_neon, float, 4);
TEST_ADD(arm_neon, int32_t, 4);

TEST_SUBTRACT(arm_neon, uint8_t, 16);
TEST_SUBTRACT(arm_neon, int8_t, 16);
TEST_SUBTRACT(arm_neon, float, 4);
TEST_SUBTRACT(arm_neon, int32_t, 4);

TEST_MULTIPLY(arm_neon, uint8_t, 16);
TEST_MULTIPLY(arm_neon, int8_t, 16);
TEST_MULTIPLY(arm_neon, float, 4);
TEST_MULTIPLY(arm_neon, int32_t, 4);

TEST_MIN(arm_neon, uint8_t, 16);
TEST_MIN(arm_neon, int8_t, 16);
TEST_MIN(arm_neon, int16_t, 8);
TEST_MIN(arm_neon, float, 4);
TEST_MIN(arm_neon, int32_t, 4);

TEST_MAX(arm_neon, uint8_t, 16);
TEST_MAX(arm_neon, int8_t, 16);
TEST_MAX(arm_neon, int16_t, 8);
TEST_MAX(arm_neon, float, 4);
TEST_MAX(arm_neon, int32_t, 4);

TEST_HORIZONTAL_MIN(arm_neon, uint8_t, 16);
TEST_HORIZONTAL_MIN(arm_neon, int8_t, 16);
TEST_HORIZONTAL_MIN(arm_neon, int16_t, 8);
TEST_HORIZONTAL_MIN(arm_neon, float, 4);
TEST_HORIZONTAL_MIN(arm_neon, int32_t, 4);

TEST_HORIZONTAL_MAX(arm_neon, uint8_t, 16);
TEST_HORIZONTAL_MAX(arm_neon, int8_t, 16);
TEST_HORIZONTAL_MAX(arm_neon, int16_t, 8);
TEST_HORIZONTAL_MAX(arm_neon, float, 4);
TEST_HORIZONTAL_MAX(arm_neon, int32_t, 4);

TEST_CONVERT(arm_neon, int32_t, s8x16);
TEST_CONVERT(arm_neon, int32_t, u8x16);
TEST_CONVERT(arm_neon, int32_t, s16x8);
TEST_CONVERT(arm_neon, float, bf16x8);

}  // namespace simd
}  // namespace ynn
