// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/hvx.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class hvx : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::hvx)) {
      GTEST_SKIP() << "HVX not supported";
    }
  }
};

TEST_BROADCAST(hvx, uint8_t, 128);
TEST_BROADCAST(hvx, int8_t, 128);
TEST_BROADCAST(hvx, int16_t, 64);
TEST_BROADCAST(hvx, half, 64);
TEST_BROADCAST(hvx, bfloat16, 64);
TEST_BROADCAST(hvx, float, 32);
TEST_BROADCAST(hvx, int32_t, 32);

TEST_LOAD_STORE(hvx, uint8_t, 128);
TEST_LOAD_STORE(hvx, int8_t, 128);
TEST_LOAD_STORE(hvx, int16_t, 64);
TEST_LOAD_STORE(hvx, half, 64);
TEST_LOAD_STORE(hvx, bfloat16, 64);
TEST_LOAD_STORE(hvx, float, 32);
TEST_LOAD_STORE(hvx, int32_t, 32);

TEST_ALIGNED_LOAD_STORE(hvx, uint8_t, 128);
TEST_ALIGNED_LOAD_STORE(hvx, int8_t, 128);
TEST_ALIGNED_LOAD_STORE(hvx, int16_t, 64);
TEST_ALIGNED_LOAD_STORE(hvx, half, 64);
TEST_ALIGNED_LOAD_STORE(hvx, bfloat16, 64);
TEST_ALIGNED_LOAD_STORE(hvx, float, 32);
TEST_ALIGNED_LOAD_STORE(hvx, int32_t, 32);

TEST_PARTIAL_LOAD_STORE(hvx, uint8_t, 128);
TEST_PARTIAL_LOAD_STORE(hvx, int8_t, 128);
TEST_PARTIAL_LOAD_STORE(hvx, int16_t, 64);
TEST_PARTIAL_LOAD_STORE(hvx, half, 64);
TEST_PARTIAL_LOAD_STORE(hvx, bfloat16, 64);
TEST_PARTIAL_LOAD_STORE(hvx, float, 32);
TEST_PARTIAL_LOAD_STORE(hvx, int32_t, 32);

TEST_ADD(hvx, uint8_t, 128);
TEST_ADD(hvx, int8_t, 128);
TEST_ADD(hvx, int16_t, 64);
TEST_ADD(hvx, int32_t, 32);

TEST_SUBTRACT(hvx, uint8_t, 128);
TEST_SUBTRACT(hvx, int8_t, 128);
TEST_SUBTRACT(hvx, int16_t, 64);
TEST_SUBTRACT(hvx, int32_t, 32);

TEST_MIN(hvx, uint8_t, 128);
TEST_MIN(hvx, int8_t, 128);
TEST_MIN(hvx, int16_t, 64);
TEST_MIN(hvx, half, 64);
TEST_MIN(hvx, float, 32);

TEST_MAX(hvx, uint8_t, 128);
TEST_MAX(hvx, int8_t, 128);
TEST_MAX(hvx, int16_t, 64);
TEST_MAX(hvx, half, 64);
TEST_MAX(hvx, float, 32);

TEST_HORIZONTAL_SUM(hvx, float, 32);
TEST_HORIZONTAL_SUM(hvx, int32_t, 32);

TEST_HORIZONTAL_MIN(hvx, uint8_t, 128);
// TODO(dsharlet): HVX vectors are so big that our test for this overflows...
// TEST_HORIZONTAL_MIN(hvx, int8_t, 128);
TEST_HORIZONTAL_MIN(hvx, int16_t, 64);
TEST_HORIZONTAL_MIN(hvx, int32_t, 32);
TEST_HORIZONTAL_MIN(hvx, half, 64);
TEST_HORIZONTAL_MIN(hvx, float, 32);

TEST_HORIZONTAL_MAX(hvx, uint8_t, 128);
// TODO(dsharlet): HVX vectors are so big that our test for this overflows...
// TEST_HORIZONTAL_MAX(hvx, int8_t, 128);
TEST_HORIZONTAL_MAX(hvx, int16_t, 64);
TEST_HORIZONTAL_MAX(hvx, int32_t, 32);
TEST_HORIZONTAL_MAX(hvx, half, 64);
TEST_HORIZONTAL_MAX(hvx, float, 32);

TEST_CONVERT(hvx, int32_t, u8x128);
TEST_CONVERT(hvx, int32_t, s8x128);
TEST_CONVERT(hvx, int32_t, s16x64);
TEST_CONVERT(hvx, int16_t, u8x128);
TEST_CONVERT(hvx, int16_t, s8x128);

}  // namespace simd
}  // namespace ynn
