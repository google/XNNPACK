// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if defined(SIMD_HEADER)
#include SIMD_HEADER
#else
#error "SIMD_HEADER is not defined."

// The following `#include`s are needed to trick some compilers and/or build
// systems into pulling in the correct SIMD header defined above.
#include "xnnpack/simd/f32-avx-base.h"
#include "xnnpack/simd/f32-avx.h"
#include "xnnpack/simd/f32-avx512f.h"
#include "xnnpack/simd/f32-fma3.h"
#include "xnnpack/simd/f32-neon.h"
#include "xnnpack/simd/f32-scalar.h"
#include "xnnpack/simd/f32-sse2.h"
#include "xnnpack/simd/f32-wasmsimd.h"
#endif  // defined(SIMD_HEADER)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "replicable_random_device.h"

namespace xnnpack {

class F32SimdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    inputs_.resize(3 * xnn_simd_size_f32);
    output_.resize(xnn_simd_size_f32);
    std::uniform_real_distribution<float> f32dist(-10.0f, 10.0f);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return f32dist(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<float> inputs_;
  std::vector<float> output_;
};

TEST_F(F32SimdTest, SetZero) {
  xnn_storeu_f32(output_.data(), xnn_zero_f32());
  EXPECT_THAT(output_, testing::Each(testing::Eq(0.0f)));
}

TEST_F(F32SimdTest, Add) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_add_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] + inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdTest, Mul) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_mul_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] * inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdTest, Fmadd) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t c =
      xnn_loadu_f32(inputs_.data() + 2 * xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_fmadd_f32(a, b, c);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] * inputs_[k + xnn_simd_size_f32] +
                              inputs_[k + 2 * xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdTest, Fnmadd) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t c =
      xnn_loadu_f32(inputs_.data() + 2 * xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_fnmadd_f32(a, b, c);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], -(inputs_[k] * inputs_[k + xnn_simd_size_f32]) +
                              inputs_[k + 2 * xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdTest, Sub) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_sub_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] - inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdTest, Div) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_div_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] / inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdTest, Max) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_max_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::max(inputs_[k], inputs_[k + xnn_simd_size_f32]));
  }
}

TEST_F(F32SimdTest, Min) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_min_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::min(inputs_[k], inputs_[k + xnn_simd_size_f32]));
  }
}

TEST_F(F32SimdTest, Abs) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t res = xnn_abs_f32(a);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::abs(inputs_[k]));
  }
}

TEST_F(F32SimdTest, Neg) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t res = xnn_neg_f32(a);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], -inputs_[k]);
  }
}

TEST_F(F32SimdTest, And) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_and_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(*(uint32_t *)&output_[k],
              *(uint32_t *)&inputs_[k] &
                  *(uint32_t *)&inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdTest, Or) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_or_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(*(uint32_t *)&output_[k],
              *(uint32_t *)&inputs_[k] |
                  *(uint32_t *)&inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdTest, Xor) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_xor_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(*(uint32_t *)&output_[k],
              *(uint32_t *)&inputs_[k] ^
                  *(uint32_t *)&inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdTest, ShiftLeft) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t res = xnn_sll_f32(a, 5);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 5);
  }
}

TEST_F(F32SimdTest, ShiftRight) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t res = xnn_srl_f32(a, 5);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 5);
  }
}

TEST_F(F32SimdTest, GetExp) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t res = xnn_getexp_f32(a);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::logb(inputs_[k]));
  }
}

}  // namespace xnnpack
