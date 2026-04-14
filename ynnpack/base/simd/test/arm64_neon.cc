// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/arm_neon.h"
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

TEST_BROADCAST(arm_neon, f64, 2);

TEST_LOAD_STORE(arm_neon, f64, 2);
TEST_ALIGNED_LOAD_STORE(arm_neon, f64, 2);
TEST_PARTIAL_LOAD_STORE(arm_neon, f64, 2);

TEST_ADD(arm_neon, f64, 2);
TEST_SUBTRACT(arm_neon, f64, 2);
TEST_MULTIPLY(arm_neon, f64, 2);
TEST_DIVIDE(arm_neon, f64, 2);
TEST_MIN(arm_neon, f64, 2);
TEST_MAX(arm_neon, f64, 2);
TEST_FLOOR(arm_neon, f64, 2);
TEST_CEIL(arm_neon, f64, 2);
TEST_ROUND(arm_neon, f64, 2);
TEST_SQRT(arm_neon, f64, 2);
TEST_ABS(arm_neon, f64, 2);

TEST_HORIZONTAL_MIN(arm_neon, f64, 2);
TEST_HORIZONTAL_MAX(arm_neon, f64, 2);

}  // namespace simd
}  // namespace ynn
