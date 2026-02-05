// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neonfp16.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
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

TEST_CONVERT(arm_neonfp16, float, f16x8);

}  // namespace simd
}  // namespace ynn
