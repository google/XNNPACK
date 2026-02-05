// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx512f.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class x86_avx512f : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::avx512f)) {
      GTEST_SKIP() << "avx512f not supported on this hardware";
    }
  }
};

TEST_BROADCAST(x86_avx512f, uint8_t, 64);
TEST_BROADCAST(x86_avx512f, int8_t, 64);
TEST_BROADCAST(x86_avx512f, int16_t, 32);
TEST_BROADCAST(x86_avx512f, half, 32);
TEST_BROADCAST(x86_avx512f, bfloat16, 32);
TEST_BROADCAST(x86_avx512f, float, 16);
TEST_BROADCAST(x86_avx512f, int32_t, 16);

TEST_LOAD_STORE(x86_avx512f, uint8_t, 64);
TEST_LOAD_STORE(x86_avx512f, int8_t, 64);
TEST_LOAD_STORE(x86_avx512f, int16_t, 32);
TEST_LOAD_STORE(x86_avx512f, half, 32);
TEST_LOAD_STORE(x86_avx512f, bfloat16, 32);
TEST_LOAD_STORE(x86_avx512f, float, 16);
TEST_LOAD_STORE(x86_avx512f, int32_t, 16);

TEST_ALIGNED_LOAD_STORE(x86_avx512f, uint8_t, 64);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, int8_t, 64);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, int16_t, 32);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, half, 32);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, bfloat16, 32);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, float, 16);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, int32_t, 16);

TEST_PARTIAL_LOAD_STORE(x86_avx512f, float, 16);
TEST_PARTIAL_LOAD_STORE(x86_avx512f, int32_t, 16);

TEST_ADD(x86_avx512f, float, 16);
TEST_ADD(x86_avx512f, int32_t, 16);

TEST_SUBTRACT(x86_avx512f, float, 16);
TEST_SUBTRACT(x86_avx512f, int32_t, 16);

TEST_MULTIPLY(x86_avx512f, float, 16);
TEST_MULTIPLY(x86_avx512f, int32_t, 16);

TEST_MIN(x86_avx512f, float, 16);
TEST_MIN(x86_avx512f, int32_t, 16);

TEST_MAX(x86_avx512f, float, 16);
TEST_MAX(x86_avx512f, int32_t, 16);

TEST_FMA(x86_avx512f, float, 16);

TEST_EXTRACT(x86_avx512f, s32x16, 4);
TEST_EXTRACT(x86_avx512f, f32x16, 4);
TEST_EXTRACT(x86_avx512f, s8x64, 16);
TEST_EXTRACT(x86_avx512f, u8x64, 16);

TEST_EXTRACT(x86_avx512f, bf16x32, 16);
TEST_EXTRACT(x86_avx512f, f16x32, 16);
TEST_EXTRACT(x86_avx512f, s8x64, 32);
TEST_EXTRACT(x86_avx512f, u8x64, 32);

TEST_CONCAT(x86_avx512f, bf16x16);
TEST_CONCAT(x86_avx512f, f16x16);
TEST_CONCAT(x86_avx512f, s8x32);
TEST_CONCAT(x86_avx512f, u8x32);

TEST_CONVERT(x86_avx512f, float, bf16x16);
TEST_CONVERT(x86_avx512f, float, f16x16);

TEST_HORIZONTAL_MIN(x86_avx512f, uint8_t, 64);
TEST_HORIZONTAL_MIN(x86_avx512f, int8_t, 64);
TEST_HORIZONTAL_MIN(x86_avx512f, int16_t, 32);
TEST_HORIZONTAL_MIN(x86_avx512f, float, 16);
TEST_HORIZONTAL_MIN(x86_avx512f, int32_t, 16);

TEST_HORIZONTAL_MAX(x86_avx512f, uint8_t, 64);
TEST_HORIZONTAL_MAX(x86_avx512f, int8_t, 64);
TEST_HORIZONTAL_MAX(x86_avx512f, int16_t, 32);
TEST_HORIZONTAL_MAX(x86_avx512f, float, 16);
TEST_HORIZONTAL_MAX(x86_avx512f, int32_t, 16);

}  // namespace simd
}  // namespace ynn
