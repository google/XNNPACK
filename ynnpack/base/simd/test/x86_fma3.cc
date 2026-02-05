// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_fma3.h"

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class x86_fma3 : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::fma3)) {
      GTEST_SKIP() << "fma3 not supported on this hardware";
    }
  }
};

TEST_FMA(x86_fma3, float, 8);

}  // namespace simd
}  // namespace ynn
