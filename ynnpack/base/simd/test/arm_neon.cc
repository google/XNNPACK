// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neon.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class arm_neon : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::neon)) {
      GTEST_SKIP() << "neon not supported on this hardware";
    }
  }
};

TEST_BROADCAST(arm_neon, u8, 16);
TEST_BROADCAST(arm_neon, s8, 16);
TEST_BROADCAST(arm_neon, s16, 8);
TEST_BROADCAST(arm_neon, f16, 8);
TEST_BROADCAST(arm_neon, bf16, 8);
TEST_BROADCAST(arm_neon, f32, 4);
TEST_BROADCAST(arm_neon, s32, 4);

TEST_LOAD_STORE(arm_neon, u8, 16);
TEST_LOAD_STORE(arm_neon, s8, 16);
TEST_LOAD_STORE(arm_neon, s16, 8);
TEST_LOAD_STORE(arm_neon, f16, 8);
TEST_LOAD_STORE(arm_neon, bf16, 8);
TEST_LOAD_STORE(arm_neon, f32, 4);
TEST_LOAD_STORE(arm_neon, s32, 4);

TEST_ALIGNED_LOAD_STORE(arm_neon, u8, 16);
TEST_ALIGNED_LOAD_STORE(arm_neon, s8, 16);
TEST_ALIGNED_LOAD_STORE(arm_neon, s16, 8);
TEST_ALIGNED_LOAD_STORE(arm_neon, f16, 8);
TEST_ALIGNED_LOAD_STORE(arm_neon, bf16, 8);
TEST_ALIGNED_LOAD_STORE(arm_neon, f32, 4);
TEST_ALIGNED_LOAD_STORE(arm_neon, s32, 4);

TEST_PARTIAL_LOAD_STORE(arm_neon, u8, 16);
TEST_PARTIAL_LOAD_STORE(arm_neon, s8, 16);
TEST_PARTIAL_LOAD_STORE(arm_neon, s16, 8);
TEST_PARTIAL_LOAD_STORE(arm_neon, f16, 8);
TEST_PARTIAL_LOAD_STORE(arm_neon, bf16, 8);
TEST_PARTIAL_LOAD_STORE(arm_neon, f32, 4);
TEST_PARTIAL_LOAD_STORE(arm_neon, s32, 4);

TEST_ADD(arm_neon, u8, 16);
TEST_ADD(arm_neon, s8, 16);
TEST_ADD(arm_neon, f32, 4);
TEST_ADD(arm_neon, s32, 4);

TEST_SUBTRACT(arm_neon, u8, 16);
TEST_SUBTRACT(arm_neon, s8, 16);
TEST_SUBTRACT(arm_neon, f32, 4);
TEST_SUBTRACT(arm_neon, s32, 4);

TEST_MULTIPLY(arm_neon, u8, 16);
TEST_MULTIPLY(arm_neon, s8, 16);
TEST_MULTIPLY(arm_neon, f32, 4);
TEST_MULTIPLY(arm_neon, s32, 4);

TEST_MIN(arm_neon, u8, 16);
TEST_MIN(arm_neon, s8, 16);
TEST_MIN(arm_neon, s16, 8);
TEST_MIN(arm_neon, f32, 4);
TEST_MIN(arm_neon, s32, 4);

TEST_MAX(arm_neon, u8, 16);
TEST_MAX(arm_neon, s8, 16);
TEST_MAX(arm_neon, s16, 8);
TEST_MAX(arm_neon, f32, 4);
TEST_MAX(arm_neon, s32, 4);

TEST_HORIZONTAL_MIN(arm_neon, u8, 16);
TEST_HORIZONTAL_MIN(arm_neon, s8, 16);
TEST_HORIZONTAL_MIN(arm_neon, s16, 8);
TEST_HORIZONTAL_MIN(arm_neon, f32, 4);
TEST_HORIZONTAL_MIN(arm_neon, s32, 4);

TEST_HORIZONTAL_MAX(arm_neon, u8, 16);
TEST_HORIZONTAL_MAX(arm_neon, s8, 16);
TEST_HORIZONTAL_MAX(arm_neon, s16, 8);
TEST_HORIZONTAL_MAX(arm_neon, f32, 4);
TEST_HORIZONTAL_MAX(arm_neon, s32, 4);

TEST_CONVERT(arm_neon, s32, s8x16);
TEST_CONVERT(arm_neon, s32, u8x16);
TEST_CONVERT(arm_neon, s32, s16x8);
TEST_CONVERT(arm_neon, f32, bf16x8);

}  // namespace simd
}  // namespace ynn
