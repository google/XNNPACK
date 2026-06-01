// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/arm_vec128.h"

// This must be included last
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class arm_neonfp16arith : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::neonfp16arith)) {
      GTEST_SKIP() << "neonfp16arith not supported on this hardware";
    }
  }
};

TEST_ADD(arm_neonfp16arith, f16, 8);
TEST_SUBTRACT(arm_neonfp16arith, f16, 8);
TEST_MULTIPLY(arm_neonfp16arith, f16, 8);
#ifdef YNN_ARCH_ARM64
TEST_DIVIDE(arm_neonfp16arith, f16, 8);
#endif
TEST_MIN(arm_neonfp16arith, f16, 8);
TEST_MAX(arm_neonfp16arith, f16, 8);
TEST_FLOOR_LOG2(arm_neonfp16arith, f16, 8);
TEST_EXP2_ROUND(arm_neonfp16arith, f16, 8);

}  // namespace simd
}  // namespace ynn
