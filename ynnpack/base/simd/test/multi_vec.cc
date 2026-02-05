// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/x86_sse2.h"

// clang-format off
#include "ynnpack/base/simd/test/generic.h"  // NOLINT
// clang-format on

namespace ynn {
namespace simd {

class multi_vec : public ::testing::Test {};

TEST_BROADCAST(multi_vec, uint8_t, 32);
TEST_BROADCAST(multi_vec, int8_t, 32);
TEST_BROADCAST(multi_vec, int16_t, 16);
TEST_BROADCAST(multi_vec, half, 16);
TEST_BROADCAST(multi_vec, bfloat16, 16);
TEST_BROADCAST(multi_vec, float, 8);
TEST_BROADCAST(multi_vec, int32_t, 8);

TEST_LOAD_STORE(multi_vec, uint8_t, 32);
TEST_LOAD_STORE(multi_vec, int8_t, 32);
TEST_LOAD_STORE(multi_vec, int16_t, 16);
TEST_LOAD_STORE(multi_vec, half, 16);
TEST_LOAD_STORE(multi_vec, bfloat16, 16);
TEST_LOAD_STORE(multi_vec, float, 8);
TEST_LOAD_STORE(multi_vec, int32_t, 8);

TEST_PARTIAL_LOAD_STORE(multi_vec, uint8_t, 32);
TEST_PARTIAL_LOAD_STORE(multi_vec, int8_t, 32);
TEST_PARTIAL_LOAD_STORE(multi_vec, int16_t, 16);
TEST_PARTIAL_LOAD_STORE(multi_vec, half, 16);
TEST_PARTIAL_LOAD_STORE(multi_vec, bfloat16, 16);
TEST_PARTIAL_LOAD_STORE(multi_vec, float, 8);
TEST_PARTIAL_LOAD_STORE(multi_vec, int32_t, 8);

TEST_ADD(multi_vec, uint8_t, 32);
TEST_ADD(multi_vec, int8_t, 32);
TEST_ADD(multi_vec, int16_t, 16);
TEST_ADD(multi_vec, float, 8);
TEST_ADD(multi_vec, int32_t, 8);

TEST_MULTIPLY(multi_vec, float, 8);

TEST_EXTRACT(multi_vec, s32x8, 4);
TEST_EXTRACT(multi_vec, f32x8, 4);
TEST_EXTRACT(multi_vec, bf16x16, 8);
TEST_EXTRACT(multi_vec, f16x16, 8);
TEST_EXTRACT(multi_vec, s8x32, 16);
TEST_EXTRACT(multi_vec, u8x32, 16);

TEST_CONCAT(multi_vec, s32x4);
TEST_CONCAT(multi_vec, f32x4);
TEST_CONCAT(multi_vec, bf16x8);
TEST_CONCAT(multi_vec, f16x8);
TEST_CONCAT(multi_vec, s8x16);
TEST_CONCAT(multi_vec, u8x16);

}  // namespace simd
}  // namespace ynn
