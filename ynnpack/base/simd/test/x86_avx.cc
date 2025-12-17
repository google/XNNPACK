// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx.h"

#include <cstdint>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_BROADCAST(x86_avx, uint8_t, 32);
TEST_BROADCAST(x86_avx, int8_t, 32);
TEST_BROADCAST(x86_avx, int16_t, 16);
TEST_BROADCAST(x86_avx, half, 16);
TEST_BROADCAST(x86_avx, bfloat16, 16);
TEST_BROADCAST(x86_avx, float, 8);
TEST_BROADCAST(x86_avx, int32_t, 8);

TEST_LOAD_STORE(x86_avx, uint8_t, 32);
TEST_LOAD_STORE(x86_avx, int8_t, 32);
TEST_LOAD_STORE(x86_avx, int16_t, 16);
TEST_LOAD_STORE(x86_avx, half, 16);
TEST_LOAD_STORE(x86_avx, bfloat16, 16);
TEST_LOAD_STORE(x86_avx, float, 8);
TEST_LOAD_STORE(x86_avx, int32_t, 8);

TEST_ALIGNED_LOAD_STORE(x86_avx, uint8_t, 32);
TEST_ALIGNED_LOAD_STORE(x86_avx, int8_t, 32);
TEST_ALIGNED_LOAD_STORE(x86_avx, int16_t, 16);
TEST_ALIGNED_LOAD_STORE(x86_avx, half, 16);
TEST_ALIGNED_LOAD_STORE(x86_avx, bfloat16, 16);
TEST_ALIGNED_LOAD_STORE(x86_avx, float, 8);
TEST_ALIGNED_LOAD_STORE(x86_avx, int32_t, 8);

TEST_PARTIAL_LOAD_STORE(x86_avx, uint8_t, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx, int8_t, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx, int16_t, 16);
TEST_PARTIAL_LOAD_STORE(x86_avx, half, 16);
TEST_PARTIAL_LOAD_STORE(x86_avx, bfloat16, 16);
TEST_PARTIAL_LOAD_STORE(x86_avx, float, 8);
TEST_PARTIAL_LOAD_STORE(x86_avx, int32_t, 8);

TEST_PARTIAL_LOAD_STORE(x86_avx, uint8_t, 16);
TEST_PARTIAL_LOAD_STORE(x86_avx, int8_t, 16);
TEST_PARTIAL_LOAD_STORE(x86_avx, int16_t, 8);
TEST_PARTIAL_LOAD_STORE(x86_avx, half, 8);
TEST_PARTIAL_LOAD_STORE(x86_avx, bfloat16, 8);
TEST_PARTIAL_LOAD_STORE(x86_avx, float, 4);
TEST_PARTIAL_LOAD_STORE(x86_avx, int32_t, 4);

TEST_ADD(x86_avx, float, 8);
TEST_SUBTRACT(x86_avx, float, 8);
TEST_MULTIPLY(x86_avx, float, 8);
TEST_MIN(x86_avx, float, 8);
TEST_MAX(x86_avx, float, 8);

TEST_EXTRACT(x86_avx, s32x8, 4);
TEST_EXTRACT(x86_avx, f32x8, 4);
TEST_EXTRACT(x86_avx, bf16x16, 8);
TEST_EXTRACT(x86_avx, f16x16, 8);
TEST_EXTRACT(x86_avx, s8x32, 16);
TEST_EXTRACT(x86_avx, u8x32, 16);

TEST_CONCAT(x86_avx, s32x4);
TEST_CONCAT(x86_avx, f32x4);
TEST_CONCAT(x86_avx, bf16x8);
TEST_CONCAT(x86_avx, f16x8);
TEST_CONCAT(x86_avx, s8x16);
TEST_CONCAT(x86_avx, u8x16);

TEST_HORIZONTAL_MIN(x86_avx, float, 8);
TEST_HORIZONTAL_MAX(x86_avx, float, 8);

}  // namespace simd
}  // namespace ynn
