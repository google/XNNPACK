// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neonfma.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
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

}  // namespace simd
}  // namespace ynn
