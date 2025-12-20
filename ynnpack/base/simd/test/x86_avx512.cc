// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx512bw.h"

#include <cstdint>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_BROADCAST(x86_avx512, uint8_t, 64);
TEST_BROADCAST(x86_avx512, int8_t, 64);
TEST_BROADCAST(x86_avx512, int16_t, 32);
TEST_BROADCAST(x86_avx512, half, 32);
TEST_BROADCAST(x86_avx512, bfloat16, 32);
TEST_BROADCAST(x86_avx512, float, 16);
TEST_BROADCAST(x86_avx512, int32_t, 16);

TEST_LOAD_STORE(x86_avx512, uint8_t, 64);
TEST_LOAD_STORE(x86_avx512, int8_t, 64);
TEST_LOAD_STORE(x86_avx512, int16_t, 32);
TEST_LOAD_STORE(x86_avx512, half, 32);
TEST_LOAD_STORE(x86_avx512, bfloat16, 32);
TEST_LOAD_STORE(x86_avx512, float, 16);
TEST_LOAD_STORE(x86_avx512, int32_t, 16);

TEST_ALIGNED_LOAD_STORE(x86_avx512, uint8_t, 64);
TEST_ALIGNED_LOAD_STORE(x86_avx512, int8_t, 64);
TEST_ALIGNED_LOAD_STORE(x86_avx512, int16_t, 32);
TEST_ALIGNED_LOAD_STORE(x86_avx512, half, 32);
TEST_ALIGNED_LOAD_STORE(x86_avx512, bfloat16, 32);
TEST_ALIGNED_LOAD_STORE(x86_avx512, float, 16);
TEST_ALIGNED_LOAD_STORE(x86_avx512, int32_t, 16);

TEST_PARTIAL_LOAD_STORE(x86_avx512, uint8_t, 64);
TEST_PARTIAL_LOAD_STORE(x86_avx512, int8_t, 64);
TEST_PARTIAL_LOAD_STORE(x86_avx512, int16_t, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx512, half, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx512, bfloat16, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx512, float, 16);
TEST_PARTIAL_LOAD_STORE(x86_avx512, int32_t, 16);

TEST_ADD(x86_avx512, uint8_t, 64);
TEST_ADD(x86_avx512, int8_t, 64);
TEST_ADD(x86_avx512, float, 16);
TEST_ADD(x86_avx512, int32_t, 16);

TEST_SUBTRACT(x86_avx512, uint8_t, 64);
TEST_SUBTRACT(x86_avx512, int8_t, 64);
TEST_SUBTRACT(x86_avx512, float, 16);
TEST_SUBTRACT(x86_avx512, int32_t, 16);

TEST_MULTIPLY(x86_avx512, float, 16);
TEST_MULTIPLY(x86_avx512, int32_t, 16);

TEST_MIN(x86_avx512, uint8_t, 64);
TEST_MIN(x86_avx512, int8_t, 64);
TEST_MIN(x86_avx512, int16_t, 32);
TEST_MIN(x86_avx512, float, 16);
TEST_MIN(x86_avx512, int32_t, 16);

TEST_MAX(x86_avx512, uint8_t, 64);
TEST_MAX(x86_avx512, int8_t, 64);
TEST_MAX(x86_avx512, int16_t, 32);
TEST_MAX(x86_avx512, float, 16);
TEST_MAX(x86_avx512, int32_t, 16);

TEST_FMA(x86_avx512, float, 16);

TEST_EXTRACT(x86_avx512, s32x16, 4);
TEST_EXTRACT(x86_avx512, f32x16, 4);
TEST_EXTRACT(x86_avx512, s8x64, 16);
TEST_EXTRACT(x86_avx512, u8x64, 16);

TEST_EXTRACT(x86_avx512, bf16x32, 16);
TEST_EXTRACT(x86_avx512, f16x32, 16);
TEST_EXTRACT(x86_avx512, s8x64, 32);
TEST_EXTRACT(x86_avx512, u8x64, 32);

TEST_CONCAT(x86_avx512, bf16x16);
TEST_CONCAT(x86_avx512, f16x16);
TEST_CONCAT(x86_avx512, s8x32);
TEST_CONCAT(x86_avx512, u8x32);

TEST_CONVERT(x86_avx512, int32_t, s8x16);
TEST_CONVERT(x86_avx512, int32_t, u8x16);
TEST_CONVERT(x86_avx512, int32_t, s8x32);
TEST_CONVERT(x86_avx512, int32_t, u8x32);
TEST_CONVERT(x86_avx512, float, bf16x16);
TEST_CONVERT(x86_avx512, float, f16x16);

TEST_HORIZONTAL_MIN(x86_avx512, uint8_t, 64);
TEST_HORIZONTAL_MIN(x86_avx512, int8_t, 64);
TEST_HORIZONTAL_MIN(x86_avx512, int16_t, 32);
TEST_HORIZONTAL_MIN(x86_avx512, float, 16);
TEST_HORIZONTAL_MIN(x86_avx512, int32_t, 16);

TEST_HORIZONTAL_MAX(x86_avx512, uint8_t, 64);
TEST_HORIZONTAL_MAX(x86_avx512, int8_t, 64);
TEST_HORIZONTAL_MAX(x86_avx512, int16_t, 32);
TEST_HORIZONTAL_MAX(x86_avx512, float, 16);
TEST_HORIZONTAL_MAX(x86_avx512, int32_t, 16);

}  // namespace simd
}  // namespace ynn
