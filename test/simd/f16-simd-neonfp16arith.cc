// clang-format off
// Auto-generated file. Do not edit!
//   Template: test/simd/f16-simd.cc.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This header needs to go first for the arch test macros.
#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f16-neonfp16arith.h"
#include "test/replicable_random_device.h"

namespace xnnpack {

class F16SimdNEONFP16ARITHTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_fp16_arith);
    inputs_.resize(3 * xnn_simd_size_f16);
    output_.resize(xnn_simd_size_f16);
    std::uniform_real_distribution<float> f32dist(-10.0f, 10.0f);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return xnn_float16_from_float(f32dist(rng_)); });
  }

  static std::vector<float> ToFloat32(const std::vector<xnn_float16> &values) {
    std::vector<float> result;
    result.reserve(values.size());
    for (const xnn_float16 &value : values) {
      result.push_back(xnn_float16_to_float(value));
    }
    return result;
  }

  static float TruncToF16(float value) {
    return xnn_float16_to_float(xnn_float16_from_float(value));
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<xnn_float16> inputs_;
  std::vector<xnn_float16> output_;
};

TEST_F(F16SimdNEONFP16ARITHTest, ConstF16FromFloat) {
  XNN_SIMD_CONST_F16_FROM_FLOAT(vone, 1.125f);
  xnn_storeu_f16(output_.data(), vone);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(xnn_float16_to_float(output_[k]), 1.125f);
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, SetZero) {
  xnn_storeu_f16(output_.data(), xnn_zero_f16());
  EXPECT_THAT(ToFloat32(output_), testing::Each(testing::Eq(0.0f)));
}

TEST_F(F16SimdNEONFP16ARITHTest, Add) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_add_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k],
              TruncToF16(inputs_f32[k] + inputs_f32[k + xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Mul) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_mul_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k],
              TruncToF16(inputs_f32[k] * inputs_f32[k + xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Fmadd) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t c =
      xnn_loadu_f16(inputs_.data() + 2 * xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_fmadd_f16(a, b, c);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
#if XNN_SIMD_HAS_NATIVE_FMA
    // If an arch claims to support FMA, it better also round things correctly.
    ASSERT_EQ(output_f32[k],
              TruncToF16(inputs_f32[k] * inputs_f32[k + xnn_simd_size_f16] +
                         inputs_f32[k + 2 * xnn_simd_size_f16]));
#else
    ASSERT_EQ(output_f32[k],
              TruncToF16(TruncToF16(inputs_f32[k] *
                                    inputs_f32[k + xnn_simd_size_f16]) +
                         inputs_f32[k + 2 * xnn_simd_size_f16]));
#endif  // XNN_SIMD_HAS_NATIVE_FMA
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Fmsub) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t c =
      xnn_loadu_f16(inputs_.data() + 2 * xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_fmsub_f16(a, b, c);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
#if XNN_SIMD_HAS_NATIVE_FMA
    // If an arch claims to support FMA, it better also round things correctly.
    ASSERT_EQ(output_f32[k],
              TruncToF16(inputs_f32[k] * inputs_f32[k + xnn_simd_size_f16] -
                         inputs_f32[k + 2 * xnn_simd_size_f16]));
#else
    ASSERT_EQ(output_f32[k],
              TruncToF16(TruncToF16(inputs_f32[k] *
                                    inputs_f32[k + xnn_simd_size_f16]) -
                         inputs_f32[k + 2 * xnn_simd_size_f16]));
#endif  // XNN_SIMD_HAS_NATIVE_FMA
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Fnmadd) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t c =
      xnn_loadu_f16(inputs_.data() + 2 * xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_fnmadd_f16(a, b, c);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
#if XNN_SIMD_HAS_NATIVE_FMA
    // If an arch claims to support FMA, it better also round things correctly.
    ASSERT_EQ(output_f32[k],
              TruncToF16(-inputs_f32[k] * inputs_f32[k + xnn_simd_size_f16] +
                         inputs_f32[k + 2 * xnn_simd_size_f16]));
#else
    ASSERT_EQ(output_f32[k],
              TruncToF16(TruncToF16(-inputs_f32[k] *
                                    inputs_f32[k + xnn_simd_size_f16]) +
                         inputs_f32[k + 2 * xnn_simd_size_f16]));
#endif  // XNN_SIMD_HAS_NATIVE_FMA
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Sub) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_sub_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k],
              TruncToF16(inputs_f32[k] - inputs_f32[k + xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Div) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_div_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_NEAR(output_f32[k],
                inputs_f32[k] / inputs_f32[k + xnn_simd_size_f16],
                2 * 9.77e-04 * std::abs(output_f32[k]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Max) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_max_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k],
              std::max(inputs_f32[k], inputs_f32[k + xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Min) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_min_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k],
              std::min(inputs_f32[k], inputs_f32[k + xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Abs) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t res = xnn_abs_f16(a);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k], std::abs(inputs_f32[k]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Neg) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t res = xnn_neg_f16(a);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k], -inputs_f32[k]);
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Round) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t res = xnn_round_f16(a);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k], std::round(inputs_f32[k]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, And) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_and_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(xnn_float16_to_bits(output_[k]),
              xnn_float16_to_bits(inputs_[k]) &
                  xnn_float16_to_bits(inputs_[k + xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Or) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_or_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(xnn_float16_to_bits(output_[k]),
              xnn_float16_to_bits(inputs_[k]) |
                  xnn_float16_to_bits(inputs_[k + xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, Xor) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_xor_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(xnn_float16_to_bits(output_[k]),
              xnn_float16_to_bits(inputs_[k]) ^
                  xnn_float16_to_bits(inputs_[k + xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, ShiftLeft) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  // Not using a loop since the `bits` parameter needs to be a compile-time
  // constant, e.g. for `neon`.
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 1);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 1) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 2);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 2) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 3);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 3) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 4);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 4) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 5);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 5) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 6);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 6) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 7);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 7) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 8);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 8) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 9);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 9) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 10);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 10) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 11);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 11) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 12);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 12) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 13);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 13) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 14);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 14) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 15);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                (xnn_float16_to_bits(inputs_[k]) << 15) & 0xFFFF);
    }
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, ShiftRight) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  // Not using a loop since the `bits` parameter needs to be a compile-time
  // constant, e.g. for `neon`.
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 1);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 1);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 2);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 2);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 3);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 3);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 4);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 4);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 5);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 5);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 6);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 6);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 7);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 7);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 8);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 8);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 9);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 9);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 10);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 10);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 11);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 11);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 12);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 12);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 13);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 13);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 14);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 14);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 15);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(xnn_float16_to_bits(output_[k]),
                xnn_float16_to_bits(inputs_[k]) >> 15);
    }
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, CmpEq) {
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    if (rng_() & 1) {
      inputs_[k + xnn_simd_size_f16] = inputs_[k];
    }
  }
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_cmpeq_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(xnn_float16_to_bits(output_[k]),
              xnn_float16_to_bits(inputs_[k]) ==
                      xnn_float16_to_bits(inputs_[k + xnn_simd_size_f16])
                  ? 0xFFFF
                  : 0);
  }
}

TEST_F(F16SimdNEONFP16ARITHTest, StoreTail) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t num_elements = 1; num_elements < xnn_simd_size_f16;
      num_elements++) {
    xnn_store_tail_f16(output_.data(), a, num_elements);
    std::vector<float> output_f32 = ToFloat32(output_);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_f32[k], inputs_f32[k]);
    }
    for (size_t k = num_elements; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_f32[k], xnn_float16_from_float(0.0f));
    }
  }
}

}  // namespace xnnpack

#endif  // XNN_ARCH_ARM64
