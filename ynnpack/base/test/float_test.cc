#include <chrono>
#include <cstdlib>
#include <limits>

#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
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

TEST(half, round_trip) {
  using info = type_info<half>;
  ReplicableRandomDevice rng;
  for (auto _ : FuzzTest(std::chrono::seconds(1))) {
    const float f = random_value<float>(rng, info::min(), info::max());
    const float rounded = static_cast<float>(static_cast<half>(f));
    // We should be within half an epsilon of the original value.
    ASSERT_NEAR(f, rounded, std::abs(f) * info::epsilon() * 0.501f);
  }
}

TEST(bfloat16, round_trip) {
  using info = type_info<bfloat16>;
  ReplicableRandomDevice rng;
  const float inf = std::numeric_limits<float>::infinity();
  const float float_max = std::numeric_limits<float>::max();

  for (auto _ : FuzzTest(std::chrono::seconds(1))) {
    const float f = random_value<float>(rng);
    const float rounded = static_cast<float>(static_cast<bfloat16>(f));
    if (std::abs(rounded) == inf) {
      // This float rounded to bf16 overflows.
      ASSERT_GE(std::abs(f), float_max / bfloat16::rounding_multiplier);
    } else {
      // We should be within half an epsilon of the original value.
      ASSERT_NEAR(f, rounded, std::abs(f) * info::epsilon() * 0.501f);
    }
  }
}

TEST(is_convert_lossless, is_convert_lossless) {
  // Identity
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp32, ynn_type_fp32));
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp16, ynn_type_fp16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_bf16, ynn_type_bf16));
  EXPECT_TRUE(is_convert_lossless(ynn_type_int8, ynn_type_int8));
  EXPECT_TRUE(is_convert_lossless(ynn_type_int32, ynn_type_int32));

  // Upcasts
  EXPECT_TRUE(is_convert_lossless(ynn_type_fp16, ynn_type_fp32));
  EXPECT_TRUE(is_convert_lossless(ynn_type_bf16, ynn_type_fp32));

  // Downcasts (Lossy)
  EXPECT_FALSE(is_convert_lossless(ynn_type_fp32, ynn_type_fp16));
  EXPECT_FALSE(is_convert_lossless(ynn_type_fp32, ynn_type_bf16));

  // Cross-casts (Lossy)
  EXPECT_FALSE(is_convert_lossless(ynn_type_fp16, ynn_type_bf16));
  EXPECT_FALSE(is_convert_lossless(ynn_type_bf16, ynn_type_fp16));

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
