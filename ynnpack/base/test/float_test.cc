#include <chrono>
#include <cstdlib>
#include <limits>

#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/type.h"

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
  TypeGenerator<float> gen(info::min(), info::max());

  for (auto _ : FuzzTest(std::chrono::seconds(1))) {
    const float f = gen(rng);
    const float rounded = static_cast<float>(static_cast<half>(f));
    // We should be within half an epsilon of the original value.
    ASSERT_NEAR(f, rounded, std::abs(f) * info::epsilon() * 0.501f);
  }
}

TEST(bfloat16, round_trip) {
  using info = type_info<bfloat16>;
  ReplicableRandomDevice rng;
  TypeGenerator<float> gen;

  const float inf = std::numeric_limits<float>::infinity();
  const float float_max = std::numeric_limits<float>::max();

  for (auto _ : FuzzTest(std::chrono::seconds(1))) {
    const float f = gen(rng);
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

}  // namespace ynn
