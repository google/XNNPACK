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

struct Stats {
  int inf_count = 0;
  int nan_count = 0;
  int subnormal_count = 0;
  float min = std::numeric_limits<float>::infinity();
  float max = -std::numeric_limits<float>::infinity();
};

constexpr int kSamples = 1000000;

template <typename T, typename... Args>
Stats RunFillRandom(Args... args) {
  ReplicableRandomDevice rng;
  Stats results;
  int n = kSamples;
  constexpr int chunk_size = 1024;
  while (n >= chunk_size) {
    T data[chunk_size];
    fill_random(data, chunk_size, rng, args...);
    for (int i = 0; i < std::min(chunk_size, n); ++i) {
      float x = data[i];
      results.min = std::min(results.min, x);
      results.max = std::max(results.max, x);

      switch (std::fpclassify(x)) {
        case FP_INFINITE:
          ++results.inf_count;
          break;
        case FP_NAN:
          ++results.nan_count;
          break;
        case FP_SUBNORMAL:
          ++results.subnormal_count;
          break;
        default:
          break;
      }
    }
    n -= chunk_size;
  }
  return results;
}

template <typename T, typename... Args>
void TestFillRandomFloats(float min, float max, Args... args) {
  Stats results = RunFillRandom<T>(args...);

  EXPECT_LE(results.min, min);
  EXPECT_GE(results.max, max);

  // Don't allow more than 0.1% of samples to be infinity.
  EXPECT_LT(results.inf_count, kSamples / 1000);
  EXPECT_EQ(results.nan_count, 0);
  EXPECT_EQ(results.subnormal_count, 0);
}

template <typename T>
void TestFillRandomFloats() {
  const float min = type_info<T>::min();
  const float max = type_info<T>::max();

  // In these cases, we generate floats with a uniformly distributed exponent.
  TestFillRandomFloats<T>(min * 0.5f, max * 0.5f);
  TestFillRandomFloats<T>(min * 0.5f, max * 0.5f, min, max);

  // In these cases, we generate uniformly distributed values.
  TestFillRandomFloats<T>(max * 0.01f, max * 0.99f, 0.0f, max);
  TestFillRandomFloats<T>(min * 0.99f, min * 0.01f, min, 0.0f);
  TestFillRandomFloats<T>(-0.99f, 0.99f, -1.0f, 1.0f);
  TestFillRandomFloats<T>(-9.9f, 9.9f, -10.0f, 10.0f);
}

TEST(TypeGenerator, float) { TestFillRandomFloats<float>(); }
TEST(TypeGenerator, half) { TestFillRandomFloats<half>(); }
TEST(TypeGenerator, bfloat16) { TestFillRandomFloats<bfloat16>(); }

template <typename T, typename... Args>
void TestFillRandomInts(int min, int max, Args... args) {
  Stats results = RunFillRandom<T>(args...);

  EXPECT_EQ(results.min, min);
  EXPECT_EQ(results.max, max);
  EXPECT_EQ(results.inf_count, 0);
  EXPECT_EQ(results.nan_count, 0);
  EXPECT_EQ(results.subnormal_count, 0);
}

TEST(TypeGenerator, int8_t) {
  const int min = -128;
  const int max = 127;
  TestFillRandomInts<int8_t>(min, max);
  TestFillRandomInts<int8_t>(min, max, min, max);
  TestFillRandomInts<int8_t>(0, max, 0, max);
  TestFillRandomInts<int8_t>(min, 0, min, 0);
  TestFillRandomInts<int8_t>(-10, 10, -10, 10);
}

TEST(TypeGenerator, uint8_t) {
  const int min = 0;
  const int max = 255;
  TestFillRandomInts<uint8_t>(min, max);
  TestFillRandomInts<uint8_t>(min, max, min, max);
  TestFillRandomInts<uint8_t>(0, max, 0, max);
  TestFillRandomInts<uint8_t>(0, 100, 0, 100);
}

}  // namespace ynn
