// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/x86_vec256.h"

// This must be included last
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

class x86_f16c : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::f16c)) {
      GTEST_SKIP() << "f16c not supported on this hardware";
    }
  }
};

TEST_CAST(x86_f16c, f32, f16x8);
TEST_CAST(x86_f16c, f32, f16x16);

TEST_UNARY(x86_f16c, exp, f16, 16, std::exp, 2);
TEST_UNARY(x86_f16c, expm1, f16, 16, std::expm1, 2);
TEST_UNARY(x86_f16c, log, f16, 16, std::log, 1);
TEST_UNARY(x86_f16c, log1p, f16, 16, std::log1p, 1);
TEST_UNARY(x86_f16c, erf, f16, 16, std::erf, 1);
TEST_UNARY(x86_f16c, tanh, f16, 16, std::tanh, 1);

}  // namespace simd
}  // namespace ynn
