// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/hexagon_hvx.h"

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
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

TEST_BROADCAST(hvx, u8, 128);
TEST_BROADCAST(hvx, s8, 128);
TEST_BROADCAST(hvx, s16, 64);
TEST_BROADCAST(hvx, f16, 64);
TEST_BROADCAST(hvx, bf16, 64);
TEST_BROADCAST(hvx, f32, 32);
TEST_BROADCAST(hvx, s32, 32);

TEST_LOAD_STORE(hvx, u8, 128);
TEST_LOAD_STORE(hvx, s8, 128);
TEST_LOAD_STORE(hvx, s16, 64);
TEST_LOAD_STORE(hvx, f16, 64);
TEST_LOAD_STORE(hvx, bf16, 64);
TEST_LOAD_STORE(hvx, f32, 32);
TEST_LOAD_STORE(hvx, s32, 32);

TEST_ALIGNED_LOAD_STORE(hvx, u8, 128);
TEST_ALIGNED_LOAD_STORE(hvx, s8, 128);
TEST_ALIGNED_LOAD_STORE(hvx, s16, 64);
TEST_ALIGNED_LOAD_STORE(hvx, f16, 64);
TEST_ALIGNED_LOAD_STORE(hvx, bf16, 64);
TEST_ALIGNED_LOAD_STORE(hvx, f32, 32);
TEST_ALIGNED_LOAD_STORE(hvx, s32, 32);

TEST_PARTIAL_LOAD_STORE(hvx, u8, 128);
TEST_PARTIAL_LOAD_STORE(hvx, s8, 128);
TEST_PARTIAL_LOAD_STORE(hvx, s16, 64);
TEST_PARTIAL_LOAD_STORE(hvx, f16, 64);
TEST_PARTIAL_LOAD_STORE(hvx, bf16, 64);
TEST_PARTIAL_LOAD_STORE(hvx, f32, 32);
TEST_PARTIAL_LOAD_STORE(hvx, s32, 32);

TEST_ADD(hvx, u8, 128);
TEST_ADD(hvx, s8, 128);
TEST_ADD(hvx, s16, 64);
TEST_ADD(hvx, s32, 32);

TEST_SUBTRACT(hvx, u8, 128);
TEST_SUBTRACT(hvx, s8, 128);
TEST_SUBTRACT(hvx, s16, 64);
TEST_SUBTRACT(hvx, s32, 32);

TEST_MULTIPLY(hvx, s32, 32);

TEST_MIN(hvx, u8, 128);
TEST_MIN(hvx, s8, 128);
TEST_MIN(hvx, s16, 64);
TEST_MIN(hvx, f16, 64);
TEST_MIN(hvx, f32, 32);

TEST_MAX(hvx, u8, 128);
TEST_MAX(hvx, s8, 128);
TEST_MAX(hvx, s16, 64);
TEST_MAX(hvx, f16, 64);
TEST_MAX(hvx, f32, 32);

TEST_HORIZONTAL_SUM(hvx, s32, 32);

TEST_HORIZONTAL_MIN(hvx, u8, 128);
// TODO(dsharlet): HVX vectors are so big that our test for this overflows...
// TEST_HORIZONTAL_MIN(hvx, s8, 128);
TEST_HORIZONTAL_MIN(hvx, s16, 64);
TEST_HORIZONTAL_MIN(hvx, s32, 32);
TEST_HORIZONTAL_MIN(hvx, f16, 64);
TEST_HORIZONTAL_MIN(hvx, f32, 32);

TEST_HORIZONTAL_MAX(hvx, u8, 128);
// TODO(dsharlet): HVX vectors are so big that our test for this overflows...
// TEST_HORIZONTAL_MAX(hvx, s8, 128);
TEST_HORIZONTAL_MAX(hvx, s16, 64);
TEST_HORIZONTAL_MAX(hvx, s32, 32);
TEST_HORIZONTAL_MAX(hvx, f16, 64);
TEST_HORIZONTAL_MAX(hvx, f32, 32);

TEST_CONVERT(hvx, s32, u8x128);
TEST_CONVERT(hvx, s32, s8x128);
TEST_CONVERT(hvx, s32, s16x64);
TEST_CONVERT(hvx, s16, u8x128);
TEST_CONVERT(hvx, s16, s8x128);

}  // namespace simd
}  // namespace ynn
