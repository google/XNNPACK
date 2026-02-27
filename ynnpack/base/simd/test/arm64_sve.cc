// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm64_sve.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class arm64_sve : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::sve)) {
      GTEST_SKIP() << "sve not supported on this hardware";
    }
  }
};

TEST_PARTIAL_LOAD_STORE(arm64_sve, u8, 16);
TEST_PARTIAL_LOAD_STORE(arm64_sve, s8, 16);
TEST_PARTIAL_LOAD_STORE(arm64_sve, s16, 8);
TEST_PARTIAL_LOAD_STORE(arm64_sve, f16, 8);
TEST_PARTIAL_LOAD_STORE(arm64_sve, bf16, 8);
TEST_PARTIAL_LOAD_STORE(arm64_sve, f32, 4);
TEST_PARTIAL_LOAD_STORE(arm64_sve, s32, 4);

TEST_PARTIAL_LOAD_STORE(arm64_sve, f16, 4);
TEST_PARTIAL_LOAD_STORE(arm64_sve, u8, 8);

}  // namespace simd
}  // namespace ynn
