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

class arm_neonfma : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::neonfma)) {
      GTEST_SKIP() << "neonfma not supported on this hardware";
    }
  }
};

TEST_FMA(arm_neonfma, f32, 4);

TEST_FLOOR(arm_neonfma, f32, 4);
TEST_CEIL(arm_neonfma, f32, 4);
TEST_ROUND(arm_neonfma, f32, 4);
TEST_SQRT(arm_neonfma, f32, 4);

TEST_UNARY(arm_neonfma, exp, f32, 4, std::exp, 2);
TEST_UNARY(arm_neonfma, expm1, f32, 4, std::expm1, 3);
TEST_UNARY(arm_neonfma, log, f32, 4, std::log, 2);
TEST_UNARY(arm_neonfma, log1p, f32, 4, std::log1p, 3);

}  // namespace simd
}  // namespace ynn
