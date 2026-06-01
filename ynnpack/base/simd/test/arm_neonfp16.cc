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

class arm_neonfp16 : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::neonfp16)) {
      GTEST_SKIP() << "neonfp16 not supported on this hardware";
    }
  }
};

TEST_CAST(arm_neonfp16, f32, f16x4);
TEST_CAST(arm_neonfp16, f32, f16x8);
TEST_CAST(arm_neonfp16, f16, f32x4);
TEST_CAST(arm_neonfp16, f16, f32x8);

TEST_UNARY(arm_neonfp16, exp, f16, 8, std::exp, 2);
TEST_UNARY(arm_neonfp16, expm1, f16, 8, std::expm1, 2);
TEST_UNARY(arm_neonfp16, log, f16, 8, std::log, 1);
TEST_UNARY(arm_neonfp16, log1p, f16, 8, std::log1p, 1);
TEST_UNARY(arm_neonfp16, erf, f16, 8, std::erf, 1);
TEST_UNARY(arm_neonfp16, tanh, f16, 8, std::tanh, 1);

}  // namespace simd
}  // namespace ynn
