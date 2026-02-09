// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "ynnpack/base/simd/scalar.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class multi_vec : public ::testing::Test {};

TEST_BROADCAST(multi_vec, u8, 8);
TEST_BROADCAST(multi_vec, s8, 8);
TEST_BROADCAST(multi_vec, s16, 4);
TEST_BROADCAST(multi_vec, f16, 4);
TEST_BROADCAST(multi_vec, bf16, 4);
TEST_BROADCAST(multi_vec, f32, 2);
TEST_BROADCAST(multi_vec, s32, 2);

TEST_LOAD_STORE(multi_vec, u8, 8);
TEST_LOAD_STORE(multi_vec, s8, 8);
TEST_LOAD_STORE(multi_vec, s16, 4);
TEST_LOAD_STORE(multi_vec, f16, 4);
TEST_LOAD_STORE(multi_vec, bf16, 4);
TEST_LOAD_STORE(multi_vec, f32, 2);
TEST_LOAD_STORE(multi_vec, s32, 2);

TEST_PARTIAL_LOAD_STORE(multi_vec, u8, 8);
TEST_PARTIAL_LOAD_STORE(multi_vec, s8, 8);
TEST_PARTIAL_LOAD_STORE(multi_vec, s16, 4);
TEST_PARTIAL_LOAD_STORE(multi_vec, f16, 4);
TEST_PARTIAL_LOAD_STORE(multi_vec, bf16, 4);
TEST_PARTIAL_LOAD_STORE(multi_vec, f32, 2);
TEST_PARTIAL_LOAD_STORE(multi_vec, s32, 2);

TEST_ADD(multi_vec, u8, 8);
TEST_ADD(multi_vec, s8, 8);
TEST_ADD(multi_vec, s16, 4);
TEST_ADD(multi_vec, f32, 2);
TEST_ADD(multi_vec, s32, 2);

TEST_MULTIPLY(multi_vec, f32, 2);

using f32x4 = vec<f32, 4>;
using s32x4 = vec<s32, 4>;
using s16x8 = vec<s16, 8>;
using bf16x8 = vec<bf16, 8>;
using f16x8 = vec<f16, 8>;
using s8x16 = vec<s8, 16>;
using u8x16 = vec<u8, 16>;

TEST_EXTRACT(multi_vec, s32x4, 2);
TEST_EXTRACT(multi_vec, f32x4, 2);
TEST_EXTRACT(multi_vec, bf16x8, 4);
TEST_EXTRACT(multi_vec, f16x8, 4);
TEST_EXTRACT(multi_vec, s8x16, 8);
TEST_EXTRACT(multi_vec, u8x16, 8);

using f32x2 = vec<f32, 2>;
using s32x2 = vec<s32, 2>;
using bf16x4 = vec<bf16, 4>;
using f16x4 = vec<f16, 4>;
using s16x4 = vec<s16, 4>;
using u8x8 = vec<u8, 8>;
using s8x8 = vec<s8, 8>;

TEST_CONCAT(multi_vec, s32x2);
TEST_CONCAT(multi_vec, f32x2);
TEST_CONCAT(multi_vec, bf16x4);
TEST_CONCAT(multi_vec, f16x4);
TEST_CONCAT(multi_vec, s8x8);
TEST_CONCAT(multi_vec, u8x8);

TEST_HORIZONTAL_SUM(multi_vec, u8, 16);
TEST_HORIZONTAL_SUM(multi_vec, s8, 16);
TEST_HORIZONTAL_SUM(multi_vec, s16, 8);
TEST_HORIZONTAL_SUM(multi_vec, f32, 4);
TEST_HORIZONTAL_SUM(multi_vec, s32, 4);

TEST_HORIZONTAL_MIN(multi_vec, u8, 16);
TEST_HORIZONTAL_MIN(multi_vec, s8, 16);
TEST_HORIZONTAL_MIN(multi_vec, s16, 8);
TEST_HORIZONTAL_MIN(multi_vec, f32, 4);
TEST_HORIZONTAL_MIN(multi_vec, s32, 4);

TEST_HORIZONTAL_MAX(multi_vec, u8, 16);
TEST_HORIZONTAL_MAX(multi_vec, s8, 16);
TEST_HORIZONTAL_MAX(multi_vec, s16, 8);
TEST_HORIZONTAL_MAX(multi_vec, f32, 4);
TEST_HORIZONTAL_MAX(multi_vec, s32, 4);

}  // namespace simd
}  // namespace ynn
