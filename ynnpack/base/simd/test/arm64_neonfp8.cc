// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/arm_vec128.h"

// This must be included last
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class arm64_neonfp8 : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::neonfp8)) {
      GTEST_SKIP() << "neonfp8 not supported on this hardware";
    }
  }
};

TEST_CAST(arm64_neonfp8, f16, f8_e4m3x8);
TEST_CAST(arm64_neonfp8, bf16, f8_e4m3x8);
TEST_CAST(arm64_neonfp8, f16, f8_e5m2x8);
TEST_CAST(arm64_neonfp8, bf16, f8_e5m2x8);
TEST_CAST(arm64_neonfp8, f8_e4m3, f32x8);
TEST_CAST(arm64_neonfp8, f8_e5m2, f32x8);
TEST_CAST(arm64_neonfp8, f8_e4m3, f16x16);
TEST_CAST(arm64_neonfp8, f8_e5m2, f16x16);

}  // namespace simd
}  // namespace ynn
