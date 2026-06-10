// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>

#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/bit_cast.h"
#include "ynnpack/base/fp8.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

template <typename T>
void test_is_zero() {
  const T positive_zero = 0.0f;
  const T negative_zero = -0.0f;
  ASSERT_NE(positive_zero.to_bits(), negative_zero.to_bits());
  ASSERT_TRUE(positive_zero.is_zero());
  ASSERT_TRUE(negative_zero.is_zero());
}

TEST(half, is_zero) { test_is_zero<half>(); }
TEST(bfloat16, is_zero) { test_is_zero<bfloat16>(); }
TEST(fp8_e5m2, is_zero) { test_is_zero<fp8_e5m2>(); }
TEST(fp8_e4m3, is_zero) { test_is_zero<fp8_e4m3>(); }

TEST(half, round_trip) {
  using info = type_info<half>;
  ReplicableRandomDevice rng;
  for (auto _ : FuzzTest(std::chrono::seconds(1))) {
    const float f = random_value<float>(rng, info::min(), info::max());
    const float rounded = static_cast<float>(static_cast<half>(f));
    // We should be within half an epsilon of the original value.
    const float tolerance =
        std::max(std::abs(f), static_cast<float>(info::smallest_normal())) *
        static_cast<float>(info::epsilon()) * 0.501f;
    ASSERT_NEAR(f, rounded, tolerance);
  }
}

TEST(bfloat16, round_trip) {
  using info = type_info<bfloat16>;
  ReplicableRandomDevice rng;

  std::uniform_int_distribution<uint32_t> bits_dist{};

  for (auto _ : FuzzTest(std::chrono::seconds(1))) {
    const float f = bit_cast<float>(bits_dist(rng));
    const float rounded = static_cast<float>(static_cast<bfloat16>(f));
    if (std::isnan(f)) {
      ASSERT_TRUE(std::isnan(rounded));
    } else if (std::isinf(rounded)) {
      // This float rounded to bf16 overflows.
      ASSERT_GE(std::abs(f), bit_cast<float>(uint32_t{0x7F7F8000}));
    } else {
      // We should be within half an epsilon of the original value.
      const float tolerance =
          std::max(std::abs(f), static_cast<float>(info::smallest_normal())) *
          static_cast<float>(info::epsilon()) * 0.501f;
      ASSERT_NEAR(f, rounded, tolerance);
    }
  }
}

TEST(fp8_e5m2, round_trip) {
  for (int i = 0; i < 256; ++i) {
    uint8_t input_bits = static_cast<uint8_t>(i);
    fp8_e5m2 input = fp8_e5m2::from_bits(input_bits);
    bfloat16 f = static_cast<bfloat16>(input);
    fp8_e5m2 output = static_cast<fp8_e5m2>(f);
    if (std::isnan(static_cast<float>(f))) {
      EXPECT_TRUE((output.to_bits() & 0x7F) > 0x7C)
          << "Expected NaN for input bits: " << i;
    } else {
      EXPECT_EQ(input.to_bits(), output.to_bits())
          << "Failed for input bits: " << i
          << " (bf16 value: " << static_cast<float>(f) << ")";
    }
  }
}

TEST(fp8_e4m3, round_trip) {
  for (int i = 0; i < 256; ++i) {
    uint8_t input_bits = static_cast<uint8_t>(i);
    fp8_e4m3 input = fp8_e4m3::from_bits(input_bits);
    bfloat16 f = static_cast<bfloat16>(input);
    fp8_e4m3 output = static_cast<fp8_e4m3>(f);
    if (std::isnan(static_cast<float>(f))) {
      EXPECT_EQ(output.to_bits() & 0x7F, 0x7F)
          << "Expected NaN for input bits: " << i;
    } else {
      EXPECT_EQ(input.to_bits(), output.to_bits())
          << "Failed for input bits: " << i
          << " (bf16 value: " << static_cast<float>(f) << ")";
    }
  }
}

TEST(fp8_e5m2, fuzz_float) {
  ReplicableRandomDevice rng;
  for (auto _ : FuzzTest(std::chrono::seconds(1))) {
    const float f = random_value<float>(rng, -57344.0f, 57344.0f);
    const float rounded = static_cast<float>(static_cast<half>(fp8_e5m2(f)));
    const float tolerance =
        std::max(std::abs(f), 0.00006103515625f) * 0.25f * 0.501f;
    ASSERT_NEAR(f, rounded, tolerance);
  }
}

TEST(fp8_e4m3, fuzz_float) {
  ReplicableRandomDevice rng;
  for (auto _ : FuzzTest(std::chrono::seconds(1))) {
    const float f = random_value<float>(rng, -448.0f, 448.0f);
    const float rounded = static_cast<float>(static_cast<half>(fp8_e4m3(f)));
    const float tolerance = std::max(std::abs(f), 0.015625f) * 0.125f * 0.501f;
    ASSERT_NEAR(f, rounded, tolerance);
  }
}

TEST(is_convert_lossless, is_convert_lossless) {
  // Identity
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp32, ynn_type_fp32));
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp16, ynn_type_fp16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_bf16, ynn_type_bf16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp8_e5m2, ynn_type_fp8_e5m2));
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp8_e4m3, ynn_type_fp8_e4m3));
  EXPECT_TRUE(is_convert_lossless(ynn_type_int8, ynn_type_int8));
  EXPECT_TRUE(is_convert_lossless(ynn_type_int32, ynn_type_int32));

  // Upcasts
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp16, ynn_type_fp32));
  EXPECT_TRUE(is_convert_lossless(ynn_type_bf16, ynn_type_fp32));
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp8_e5m2, ynn_type_fp32));
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp8_e4m3, ynn_type_fp32));
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp8_e5m2, ynn_type_fp16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp8_e4m3, ynn_type_fp16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp8_e5m2, ynn_type_bf16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp8_e4m3, ynn_type_bf16));

  // Downcasts (Lossy)
  EXPECT_FALSE(is_convert_lossless(ynn_type_fp32, ynn_type_fp16));
  EXPECT_FALSE(is_convert_lossless(ynn_type_fp32, ynn_type_bf16));
  EXPECT_FALSE(is_convert_lossless(ynn_type_fp32, ynn_type_fp8_e5m2));
  EXPECT_FALSE(is_convert_lossless(ynn_type_fp32, ynn_type_fp8_e4m3));

  // Cross-casts (Lossy)
  EXPECT_FALSE(is_convert_lossless(ynn_type_fp16, ynn_type_bf16));
  EXPECT_FALSE(is_convert_lossless(ynn_type_bf16, ynn_type_fp16));
  EXPECT_FALSE(is_convert_lossless(ynn_type_fp8_e5m2, ynn_type_fp8_e4m3));
  EXPECT_FALSE(is_convert_lossless(ynn_type_fp8_e4m3, ynn_type_fp8_e5m2));

  // int8 to floats
  EXPECT_TRUE(is_convert_lossless(ynn_type_int8, ynn_type_fp16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_uint8, ynn_type_fp16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_int8, ynn_type_bf16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_uint8, ynn_type_bf16));

  EXPECT_TRUE(is_convert_lossless(ynn_type_int4, ynn_type_bf16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_uint4, ynn_type_bf16));

  // int32 to fp32 (32 bits > 23 bits)
  EXPECT_FALSE(is_convert_lossless(ynn_type_int32, ynn_type_fp32));
}

}  // namespace ynn
