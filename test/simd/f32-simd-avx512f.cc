// clang-format off
// Auto-generated file. Do not edit!
//   Template: test/simd/f32-simd.cc.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// This header needs to go first for the arch test macros.
#include "src/xnnpack/common.h"

#if XNN_ARCH_X86 || XNN_ARCH_X86_64

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/simd/f32-avx512f.h"
#include "test/replicable_random_device.h"

namespace xnnpack {

class F32SimdAVX512FTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_avx512f);
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

TEST_F(F32SimdAVX512FTest, SetZero) {
  xnn_storeu_f32(output_.data(), xnn_zero_f32());
  EXPECT_THAT(output_, testing::Each(testing::Eq(0.0f)));
}

TEST_F(F32SimdAVX512FTest, Add) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_add_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] + inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdAVX512FTest, Mul) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_mul_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] * inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdAVX512FTest, Fmadd) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t c =
      xnn_loadu_f32(inputs_.data() + 2 * xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_fmadd_f32(a, b, c);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
#if XNN_SIMD_HAS_NATIVE_FMA
    // If an arch claims to support FMA, it better also round things correctly.
    ASSERT_EQ(output_[k], std::fma(inputs_[k], inputs_[k + xnn_simd_size_f32],
                                  inputs_[k + 2 * xnn_simd_size_f32]));
#else
    ASSERT_EQ(output_[k],
              inputs_[k] * inputs_[k + xnn_simd_size_f32] +
                  inputs_[k + 2 * xnn_simd_size_f32]);
#endif  // XNN_SIMD_HAS_NATIVE_FMA
  }
}

TEST_F(F32SimdAVX512FTest, Fmsub) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t c =
      xnn_loadu_f32(inputs_.data() + 2 * xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_fmsub_f32(a, b, c);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
#if XNN_SIMD_HAS_NATIVE_FMA
    // If an arch claims to support FMA, it better also round things correctly.
    ASSERT_EQ(output_[k], std::fma(inputs_[k], inputs_[k + xnn_simd_size_f32],
                                   -inputs_[k + 2 * xnn_simd_size_f32]));
#else
    ASSERT_EQ(output_[k],
              inputs_[k] * inputs_[k + xnn_simd_size_f32] -
                  inputs_[k + 2 * xnn_simd_size_f32]);
#endif  // XNN_SIMD_HAS_NATIVE_FMA
  }
}

TEST_F(F32SimdAVX512FTest, Fnmadd) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t c =
      xnn_loadu_f32(inputs_.data() + 2 * xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_fnmadd_f32(a, b, c);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
#if XNN_SIMD_HAS_NATIVE_FMA
    // If an arch claims to support FMA, it better also round things correctly.
    ASSERT_EQ(output_[k], std::fma(-inputs_[k], inputs_[k + xnn_simd_size_f32],
                                   inputs_[k + 2 * xnn_simd_size_f32]));
#else
    ASSERT_EQ(output_[k],
              -inputs_[k] * inputs_[k + xnn_simd_size_f32] +
                  inputs_[k + 2 * xnn_simd_size_f32]);
#endif  // XNN_SIMD_HAS_NATIVE_FMA
  }
}

TEST_F(F32SimdAVX512FTest, Sub) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_sub_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] - inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdAVX512FTest, Div) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_div_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_NEAR(output_[k], inputs_[k] / inputs_[k + xnn_simd_size_f32],
    2 * std::numeric_limits<float>::epsilon() * std::abs(output_[k]));
  }
}

TEST_F(F32SimdAVX512FTest, Max) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_max_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::max(inputs_[k], inputs_[k + xnn_simd_size_f32]));
  }
}

TEST_F(F32SimdAVX512FTest, Min) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_min_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::min(inputs_[k], inputs_[k + xnn_simd_size_f32]));
  }
}

TEST_F(F32SimdAVX512FTest, Abs) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t res = xnn_abs_f32(a);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::abs(inputs_[k]));
  }
}

TEST_F(F32SimdAVX512FTest, Neg) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t res = xnn_neg_f32(a);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], -inputs_[k]);
  }
}

TEST_F(F32SimdAVX512FTest, Round) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t res = xnn_round_f32(a);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(output_[k], std::round(inputs_[k]));
  }

  // Check non-finite values.
  for (const float val : {INFINITY, -INFINITY, NAN}) {
    inputs_[0] = val;
    const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
    const xnn_simd_f32_t res = xnn_round_f32(a);
    xnn_storeu_f32(output_.data(), res);
    ASSERT_THAT(output_[0], testing::NanSensitiveFloatEq(std::round(val)));
  }
}

TEST_F(F32SimdAVX512FTest, ReduceAdd) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  float res = xnn_reduce_add_f32(a);
  float expected = 0.0f;
  float max_abs_input = 0.0f;
  for (size_t i = 0; i < xnn_simd_size_f32; ++i) {
    expected += inputs_[i];
    max_abs_input = std::max(max_abs_input, std::abs(inputs_[i]));
  }
  // Don't expect an exact result due to implementation-defined behavior.
  float tolerance =
      std::numeric_limits<float>::epsilon() * max_abs_input * xnn_simd_size_f32 * 2.0f;
  ASSERT_NEAR(res, expected, tolerance);
}

TEST_F(F32SimdAVX512FTest, ReduceMin) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  float res = xnn_reduce_min_f32(a);
  float expected = std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < xnn_simd_size_f32; ++i) {
    expected = std::min(expected, inputs_[i]);
  }
  ASSERT_EQ(res, expected);
}

TEST_F(F32SimdAVX512FTest, ReduceMax) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  float res = xnn_reduce_max_f32(a);
  float expected = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < xnn_simd_size_f32; ++i) {
    expected = std::max(expected, inputs_[i]);
  }
  ASSERT_EQ(res, expected);
}

TEST_F(F32SimdAVX512FTest, And) {
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

TEST_F(F32SimdAVX512FTest, Or) {
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

TEST_F(F32SimdAVX512FTest, Xor) {
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

TEST_F(F32SimdAVX512FTest, AndNot) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_andnot_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(*(uint32_t *)&output_[k],
              ~(*(uint32_t *)&inputs_[k]) &
                  *(uint32_t *)&inputs_[k + xnn_simd_size_f32]);
  }
}

TEST_F(F32SimdAVX512FTest, ShiftLeft) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  // Not using a loop since the `bits` parameter needs to be a compile-time
  // constant, e.g. for `neon`.
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 1);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 1);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 2);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 2);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 3);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 3);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 4);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 4);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 5);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 5);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 6);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 6);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 7);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 7);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 8);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 8);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 9);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 9);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 10);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 10);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 11);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 11);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 12);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 12);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 13);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 13);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 14);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 14);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 15);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 15);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 16);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 16);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 17);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 17);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 18);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 18);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 19);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 19);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 20);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 20);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 21);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 21);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 22);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 22);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 23);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 23);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 24);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 24);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 25);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 25);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 26);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 26);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 27);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 27);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 28);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 28);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 29);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 29);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 30);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 30);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sll_f32(a, 31);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] << 31);
    }
  }
}

TEST_F(F32SimdAVX512FTest, ShiftRight) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  // Not using a loop since the `bits` parameter needs to be a compile-time
  // constant, e.g. for `neon`.
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 1);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 1);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 2);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 2);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 3);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 3);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 4);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 4);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 5);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 5);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 6);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 6);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 7);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 7);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 8);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 8);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 9);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 9);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 10);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 10);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 11);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 11);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 12);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 12);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 13);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 13);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 14);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 14);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 15);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 15);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 16);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 16);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 17);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 17);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 18);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 18);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 19);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 19);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 20);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 20);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 21);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 21);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 22);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 22);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 23);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 23);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 24);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 24);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 25);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 25);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 26);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 26);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 27);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 27);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 28);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 28);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 29);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 29);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 30);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 30);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_srl_f32(a, 31);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(uint32_t *)&output_[k], *(uint32_t *)&inputs_[k] >> 31);
    }
  }
}

TEST_F(F32SimdAVX512FTest, ShiftRightSigned) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  // Not using a loop since the `bits` parameter needs to be a compile-time
  // constant, e.g. for `neon`.
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 1);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 1);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 2);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 2);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 3);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 3);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 4);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 4);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 5);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 5);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 6);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 6);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 7);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 7);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 8);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 8);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 9);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 9);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 10);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 10);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 11);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 11);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 12);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 12);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 13);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 13);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 14);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 14);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 15);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 15);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 16);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 16);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 17);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 17);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 18);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 18);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 19);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 19);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 20);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 20);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 21);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 21);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 22);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 22);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 23);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 23);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 24);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 24);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 25);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 25);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 26);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 26);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 27);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 27);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 28);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 28);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 29);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 29);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 30);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 30);
    }
  }
  {
    const xnn_simd_f32_t res = xnn_sra_f32(a, 31);
    xnn_storeu_f32(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(*(int32_t *)&output_[k], *(int32_t *)&inputs_[k] >> 31);
    }
  }
}

TEST_F(F32SimdAVX512FTest, CmpEq) {
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    if (rng_() & 1) {
      inputs_[k + xnn_simd_size_f32] = inputs_[k];
    }
  }
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const xnn_simd_f32_t b = xnn_loadu_f32(inputs_.data() + xnn_simd_size_f32);
  const xnn_simd_f32_t res = xnn_cmpeq_f32(a, b);
  xnn_storeu_f32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    ASSERT_EQ(*(uint32_t *)&output_[k],
              inputs_[k] == inputs_[k + xnn_simd_size_f32] ? 0xFFFFFFFF : 0);
  }
}

#if XNN_SIMD_HAVE_REDUCE_ADD_F32
TEST_F(F32SimdAVX512FTest, Reduce_Add) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  const float res = xnn_reduce_add_f32(a);
  float sum = 0.0f;
  for (size_t k = 0; k < xnn_simd_size_f32; k++) {
    sum += inputs_[k];
  }
  const float tolerance = std::numeric_limits<float>::epsilon() * std::abs(sum);
  ASSERT_NEAR(res, sum, tolerance);
}
#endif

TEST_F(F32SimdAVX512FTest, StoreTail) {
  const xnn_simd_f32_t a = xnn_loadu_f32(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_f32;
      num_elements++) {
    xnn_store_tail_f32(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]);
    }
    for (size_t k = num_elements; k < xnn_simd_size_f32; k++) {
      ASSERT_EQ(output_[k], 0.0f);
    }
  }
}

}  // namespace xnnpack

#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
