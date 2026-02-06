// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/scalar.h"

// clang-format off
#include "ynnpack/base/simd/test/generic.h"  // NOLINT
// clang-format on

namespace ynn {
namespace simd {

class multi_vec : public ::testing::Test {};

TEST_BROADCAST(multi_vec, uint8_t, 8);
TEST_BROADCAST(multi_vec, int8_t, 8);
TEST_BROADCAST(multi_vec, int16_t, 4);
TEST_BROADCAST(multi_vec, half, 4);
TEST_BROADCAST(multi_vec, bfloat16, 4);
TEST_BROADCAST(multi_vec, float, 2);
TEST_BROADCAST(multi_vec, int32_t, 2);

TEST_LOAD_STORE(multi_vec, uint8_t, 8);
TEST_LOAD_STORE(multi_vec, int8_t, 8);
TEST_LOAD_STORE(multi_vec, int16_t, 4);
TEST_LOAD_STORE(multi_vec, half, 4);
TEST_LOAD_STORE(multi_vec, bfloat16, 4);
TEST_LOAD_STORE(multi_vec, float, 2);
TEST_LOAD_STORE(multi_vec, int32_t, 2);

TEST_PARTIAL_LOAD_STORE(multi_vec, uint8_t, 8);
TEST_PARTIAL_LOAD_STORE(multi_vec, int8_t, 8);
TEST_PARTIAL_LOAD_STORE(multi_vec, int16_t, 4);
TEST_PARTIAL_LOAD_STORE(multi_vec, half, 4);
TEST_PARTIAL_LOAD_STORE(multi_vec, bfloat16, 4);
TEST_PARTIAL_LOAD_STORE(multi_vec, float, 2);
TEST_PARTIAL_LOAD_STORE(multi_vec, int32_t, 2);

TEST_ADD(multi_vec, uint8_t, 8);
TEST_ADD(multi_vec, int8_t, 8);
TEST_ADD(multi_vec, int16_t, 4);
TEST_ADD(multi_vec, float, 2);
TEST_ADD(multi_vec, int32_t, 2);

TEST_MULTIPLY(multi_vec, float, 2);

using f32x4 = vec<float, 4>;
using s32x4 = vec<int32_t, 4>;
using s16x8 = vec<int16_t, 8>;
using bf16x8 = vec<bfloat16, 8>;
using f16x8 = vec<half, 8>;
using s8x16 = vec<int8_t, 16>;
using u8x16 = vec<uint8_t, 16>;

TEST_EXTRACT(multi_vec, s32x4, 2);
TEST_EXTRACT(multi_vec, f32x4, 2);
TEST_EXTRACT(multi_vec, bf16x8, 4);
TEST_EXTRACT(multi_vec, f16x8, 4);
TEST_EXTRACT(multi_vec, s8x16, 8);
TEST_EXTRACT(multi_vec, u8x16, 8);

using f32x2 = vec<float, 2>;
using s32x2 = vec<int32_t, 2>;
using bf16x4 = vec<bfloat16, 4>;
using f16x4 = vec<half, 4>;
using s16x4 = vec<int16_t, 4>;
using u8x8 = vec<uint8_t, 8>;
using s8x8 = vec<int8_t, 8>;

TEST_CONCAT(multi_vec, s32x2);
TEST_CONCAT(multi_vec, f32x2);
TEST_CONCAT(multi_vec, bf16x4);
TEST_CONCAT(multi_vec, f16x4);
TEST_CONCAT(multi_vec, s8x8);
TEST_CONCAT(multi_vec, u8x8);

TEST_HORIZONTAL_SUM(multi_vec, uint8_t, 16);
TEST_HORIZONTAL_SUM(multi_vec, int8_t, 16);
TEST_HORIZONTAL_SUM(multi_vec, int16_t, 8);
TEST_HORIZONTAL_SUM(multi_vec, float, 4);
TEST_HORIZONTAL_SUM(multi_vec, int32_t, 4);

TEST_HORIZONTAL_MIN(multi_vec, uint8_t, 16);
TEST_HORIZONTAL_MIN(multi_vec, int8_t, 16);
TEST_HORIZONTAL_MIN(multi_vec, int16_t, 8);
TEST_HORIZONTAL_MIN(multi_vec, float, 4);
TEST_HORIZONTAL_MIN(multi_vec, int32_t, 4);

TEST_HORIZONTAL_MAX(multi_vec, uint8_t, 16);
TEST_HORIZONTAL_MAX(multi_vec, int8_t, 16);
TEST_HORIZONTAL_MAX(multi_vec, int16_t, 8);
TEST_HORIZONTAL_MAX(multi_vec, float, 4);
TEST_HORIZONTAL_MAX(multi_vec, int32_t, 4);

}  // namespace simd
}  // namespace ynn
