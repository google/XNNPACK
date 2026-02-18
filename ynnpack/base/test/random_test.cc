// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/test/random.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/type.h"

namespace ynn {

struct TypeGeneratorResults {
  int inf_count = 0;
  float min = std::numeric_limits<float>::infinity();
  float max = -std::numeric_limits<float>::infinity();
};

constexpr int kSamples = 1000000;

template <typename Gen>
TypeGeneratorResults RunTypeGenerator(Gen gen) {
  ReplicableRandomDevice rng;
  TypeGeneratorResults results;
  for (int i = 0; i < kSamples; ++i) {
    const float x = gen(rng);
    results.min = std::min(results.min, x);
    results.max = std::max(results.max, x);
    if (std::isinf(x)) ++results.inf_count;
  }
  return results;
}

template <typename T>
void TestFloatGenerator(TypeGenerator<T> gen, float min, float max) {
  TypeGeneratorResults results = RunTypeGenerator(gen);

  EXPECT_LE(results.min, min);
  EXPECT_GE(results.max, max);

  // Don't allow more than 0.1% of samples to be infinity.
  EXPECT_LT(results.inf_count, kSamples / 1000);
}

template <typename T>
void TestFloatGenerator() {
  const float min = -type_info<T>::max();
  const float max = type_info<T>::max();

  // In these cases, we generate floats with a uniformly distributed exponent.
  TestFloatGenerator<T>(TypeGenerator<T>{}, min * 0.5f, max * 0.5f);
  TestFloatGenerator<T>(TypeGenerator<T>{min, max}, min * 0.5f, max * 0.5f);

  // In these cases, we generate uniformly distributed values.
  TestFloatGenerator<T>(TypeGenerator<T>{0.0f, max}, max * 0.01f, max * 0.99f);
  TestFloatGenerator<T>(TypeGenerator<T>{min, 0.0f}, min * 0.99f, min * 0.01f);
  TestFloatGenerator<T>(TypeGenerator<T>{-1.0f, 1.0f}, -0.99f, 0.99f);
  TestFloatGenerator<T>(TypeGenerator<T>{-10.0f, 10.0f}, -9.9f, 9.9f);
}

TEST(TypeGenerator, float) { TestFloatGenerator<float>(); }
TEST(TypeGenerator, half) { TestFloatGenerator<half>(); }
TEST(TypeGenerator, bfloat16) { TestFloatGenerator<bfloat16>(); }

template <typename T>
void TestIntGenerator(TypeGenerator<T> gen, int min, int max) {
  TypeGeneratorResults results = RunTypeGenerator(gen);

  EXPECT_LE(results.min, min);
  EXPECT_GE(results.max, max);
  EXPECT_EQ(results.inf_count, 0);
}

TEST(TypeGenerator, int8_t) {
  const int min = -128;
  const int max = 127;
  TestIntGenerator<int8_t>(TypeGenerator<int8_t>{}, min, max);
  TestIntGenerator<int8_t>(TypeGenerator<int8_t>{min, max}, min, max);
  TestIntGenerator<int8_t>(TypeGenerator<int8_t>{0, max}, 0, max);
  TestIntGenerator<int8_t>(TypeGenerator<int8_t>{min, 0}, min, 0);
  TestIntGenerator<int8_t>(TypeGenerator<int8_t>{-10, 10}, -10, 10);
}

TEST(TypeGenerator, uint8_t) {
  const int min = 0;
  const int max = 255;
  TestIntGenerator<uint8_t>(TypeGenerator<uint8_t>{}, min, max);
  TestIntGenerator<uint8_t>(TypeGenerator<uint8_t>{min, max}, min, max);
  TestIntGenerator<uint8_t>(TypeGenerator<uint8_t>{0, 10}, 0, 10);
}

}  // namespace ynn
