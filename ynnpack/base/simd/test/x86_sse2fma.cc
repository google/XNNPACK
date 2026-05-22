// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>

#include <gtest/gtest.h>
#include "ynnpack/base/simd/x86_vec128.h"

// This must be included last
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

// We assume we can always use sse2.
class x86_sse2fma : public ::testing::Test {};

TEST_FMA(x86_sse2fma, f32, 4);

TEST_UNARY(x86_sse2fma, exp, f32, 4, std::exp, 3);
TEST_UNARY(x86_sse2fma, expm1, f32, 4, std::expm1, 3);
TEST_UNARY(x86_sse2fma, log, f32, 4, std::log, 3);
TEST_UNARY(x86_sse2fma, log1p, f32, 4, std::log1p, 3);

}  // namespace simd
}  // namespace ynn
