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

TEST_PARTIAL_LOAD_STORE(x86_avx512bw, uint8_t, 64);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, int8_t, 64);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, int16_t, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, half, 32);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, bfloat16, 32);

TEST_ADD(x86_avx512bw, uint8_t, 64);
TEST_ADD(x86_avx512bw, int8_t, 64);

TEST_SUBTRACT(x86_avx512bw, uint8_t, 64);
TEST_SUBTRACT(x86_avx512bw, int8_t, 64);

TEST_MULTIPLY(x86_avx512bw, float, 16);
TEST_MULTIPLY(x86_avx512bw, int32_t, 16);

TEST_MIN(x86_avx512bw, uint8_t, 64);
TEST_MIN(x86_avx512bw, int8_t, 64);
TEST_MIN(x86_avx512bw, int16_t, 32);

TEST_MAX(x86_avx512bw, uint8_t, 64);
TEST_MAX(x86_avx512bw, int8_t, 64);
TEST_MAX(x86_avx512bw, int16_t, 32);

TEST_CONVERT(x86_avx512bw, int32_t, s8x16);
TEST_CONVERT(x86_avx512bw, int32_t, u8x16);
TEST_CONVERT(x86_avx512bw, int32_t, s8x32);
TEST_CONVERT(x86_avx512bw, int32_t, u8x32);

}  // namespace simd
}  // namespace ynn
