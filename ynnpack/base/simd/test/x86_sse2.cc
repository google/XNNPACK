// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse2.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

// We assume we can always use sse2.
class x86_sse2 : public ::testing::Test {};

TEST_BROADCAST(x86_sse2, uint8_t, 16);
TEST_BROADCAST(x86_sse2, int8_t, 16);
TEST_BROADCAST(x86_sse2, int16_t, 8);
TEST_BROADCAST(x86_sse2, half, 8);
TEST_BROADCAST(x86_sse2, bfloat16, 8);
TEST_BROADCAST(x86_sse2, float, 4);
TEST_BROADCAST(x86_sse2, int32_t, 4);

TEST_LOAD_STORE(x86_sse2, uint8_t, 16);
TEST_LOAD_STORE(x86_sse2, int8_t, 16);
TEST_LOAD_STORE(x86_sse2, int16_t, 8);
TEST_LOAD_STORE(x86_sse2, half, 8);
TEST_LOAD_STORE(x86_sse2, bfloat16, 8);
TEST_LOAD_STORE(x86_sse2, float, 4);
TEST_LOAD_STORE(x86_sse2, int32_t, 4);

TEST_ALIGNED_LOAD_STORE(x86_sse2, uint8_t, 16);
TEST_ALIGNED_LOAD_STORE(x86_sse2, int8_t, 16);
TEST_ALIGNED_LOAD_STORE(x86_sse2, int16_t, 8);
TEST_ALIGNED_LOAD_STORE(x86_sse2, half, 8);
TEST_ALIGNED_LOAD_STORE(x86_sse2, bfloat16, 8);
TEST_ALIGNED_LOAD_STORE(x86_sse2, float, 4);
TEST_ALIGNED_LOAD_STORE(x86_sse2, int32_t, 4);

TEST_PARTIAL_LOAD_STORE(x86_sse2, uint8_t, 16);
TEST_PARTIAL_LOAD_STORE(x86_sse2, int8_t, 16);
TEST_PARTIAL_LOAD_STORE(x86_sse2, int16_t, 8);
TEST_PARTIAL_LOAD_STORE(x86_sse2, half, 8);
TEST_PARTIAL_LOAD_STORE(x86_sse2, bfloat16, 8);
TEST_PARTIAL_LOAD_STORE(x86_sse2, float, 4);
TEST_PARTIAL_LOAD_STORE(x86_sse2, int32_t, 4);

TEST_ADD(x86_sse2, uint8_t, 16);
TEST_ADD(x86_sse2, int8_t, 16);
TEST_ADD(x86_sse2, int16_t, 8);
TEST_ADD(x86_sse2, float, 4);
TEST_ADD(x86_sse2, int32_t, 4);

TEST_SUBTRACT(x86_sse2, uint8_t, 16);
TEST_SUBTRACT(x86_sse2, int8_t, 16);
TEST_SUBTRACT(x86_sse2, int16_t, 8);
TEST_SUBTRACT(x86_sse2, float, 4);
TEST_SUBTRACT(x86_sse2, int32_t, 4);

TEST_MULTIPLY(x86_sse2, float, 4);

TEST_MIN(x86_sse2, uint8_t, 16);
TEST_MIN(x86_sse2, int16_t, 8);
TEST_MIN(x86_sse2, float, 4);

TEST_MAX(x86_sse2, uint8_t, 16);
TEST_MAX(x86_sse2, int16_t, 8);
TEST_MAX(x86_sse2, float, 4);

TEST_HORIZONTAL_MIN(x86_sse2, uint8_t, 16);
TEST_HORIZONTAL_MIN(x86_sse2, int16_t, 8);
TEST_HORIZONTAL_MIN(x86_sse2, float, 4);

TEST_HORIZONTAL_MAX(x86_sse2, uint8_t, 16);
TEST_HORIZONTAL_MAX(x86_sse2, int16_t, 8);
TEST_HORIZONTAL_MAX(x86_sse2, float, 4);

TEST_CONVERT(x86_sse2, float, bf16x8);
TEST_CONVERT(x86_sse2, int32_t, u8x16);
TEST_CONVERT(x86_sse2, int32_t, s8x16);

}  // namespace simd
}  // namespace ynn
