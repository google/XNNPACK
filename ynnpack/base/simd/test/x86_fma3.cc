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

class x86_fma3 : public ::testing::Test {
  void SetUp() override {
    if (!is_arch_supported(arch_flag::fma3)) {
      GTEST_SKIP() << "fma3 not supported on this hardware";
    }
  }
};

TEST_FMA(x86_fma3, f32, 8);

TEST_UNARY(x86_fma3, exp, f32, 8, std::exp, 2);
TEST_UNARY(x86_fma3, exp, f64, 4, std::exp, 2);
TEST_UNARY(x86_fma3, expm1, f32, 8, std::expm1, 2);
TEST_UNARY(x86_fma3, expm1, f64, 4, std::expm1, 2);
TEST_UNARY(x86_fma3, log, f32, 8, std::log, 2);
TEST_UNARY(x86_fma3, log, f64, 4, std::log, 2);
TEST_UNARY(x86_fma3, log1p, f32, 8, std::log1p, 3);
TEST_UNARY(x86_fma3, log1p, f64, 4, std::log1p, 3);
TEST_UNARY(x86_fma3, erf, f32, 8, std::erf, 2);
TEST_UNARY(x86_fma3, erf, f64, 4, std::erf, 3);
TEST_UNARY(x86_fma3, tanh, f32, 8, std::tanh, 5);
TEST_UNARY(x86_fma3, tanh, f64, 4, std::tanh, 4);

TEST_UNARY(x86_fma3, fast_erf, f32, 8, std::erf, 5);

}  // namespace simd
}  // namespace ynn
