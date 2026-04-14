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

class arm_neonbf16 : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::neonbf16)) {
      GTEST_SKIP() << "neonbf16 not supported on this hardware";
    }
  }
};

TEST_CAST(arm_neonbf16, bf16, f32x8);

}  // namespace simd
}  // namespace ynn
