// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx2.h"

#include <cstdint>

#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_ADD(x86_avx2, uint8_t, 32);
TEST_ADD(x86_avx2, int8_t, 32);
TEST_ADD(x86_avx2, int32_t, 8);

TEST_SUBTRACT(x86_avx2, uint8_t, 32);
TEST_SUBTRACT(x86_avx2, int8_t, 32);
TEST_SUBTRACT(x86_avx2, int32_t, 8);

TEST_MULTIPLY(x86_avx2, int32_t, 8);

TEST_MIN(x86_avx2, uint8_t, 32);
TEST_MIN(x86_avx2, int8_t, 32);
TEST_MIN(x86_avx2, int16_t, 16);
TEST_MIN(x86_avx2, int32_t, 8);

TEST_MAX(x86_avx2, uint8_t, 32);
TEST_MAX(x86_avx2, int8_t, 32);
TEST_MAX(x86_avx2, int16_t, 16);
TEST_MAX(x86_avx2, int32_t, 8);

TEST_HORIZONTAL_MIN(x86_avx2, uint8_t, 32);
TEST_HORIZONTAL_MIN(x86_avx2, int8_t, 32);
TEST_HORIZONTAL_MIN(x86_avx2, int16_t, 16);
TEST_HORIZONTAL_MIN(x86_avx2, int32_t, 8);

TEST_HORIZONTAL_MAX(x86_avx2, uint8_t, 32);
TEST_HORIZONTAL_MAX(x86_avx2, int8_t, 32);
TEST_HORIZONTAL_MAX(x86_avx2, int16_t, 16);
TEST_HORIZONTAL_MAX(x86_avx2, int32_t, 8);

TEST_CONVERT(x86_avx2, float, bf16x8);
TEST_CONVERT(x86_avx2, int32_t, u8x16);
TEST_CONVERT(x86_avx2, int32_t, s8x16);

}  // namespace simd
}  // namespace ynn
