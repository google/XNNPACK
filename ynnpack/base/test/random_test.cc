// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/test/random.h"

#include <cmath>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"

namespace ynn {

TEST(TypeGenerator, Float) {
  ReplicableRandomDevice rng;

  int inf_count = 0;
  constexpr int kSamples = 1000000;
  TypeGenerator<float> gen;
  for (int i = 0; i < kSamples; ++i) {
    float x = gen(rng);
    if (std::isinf(x)) ++inf_count;
  }
  // Don't allow more than 0.1% of samples to be infinity.
  ASSERT_LT(inf_count, kSamples / 1000);
}

}  // namespace ynn
