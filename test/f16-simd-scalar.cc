// Auto-generated file. Do not edit!
//   Template: test/f16-simd.cc.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack/isa-checks.h"
#include "xnnpack/simd/f16-scalar.h"
#include "replicable_random_device.h"

namespace xnnpack {

class F16SimdSCALARTest : public ::testing::Test {
 protected:
  void SetUp() override {
    inputs_.resize(3 * xnn_simd_size_f16);
    output_.resize(xnn_simd_size_f16);
    std::uniform_real_distribution<float> f32dist(-10.0f, 10.0f);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return fp16_ieee_from_fp32_value(f32dist(rng_)); });
  }

  static std::vector<float> ToFloat32(const std::vector<uint16_t> &values) {
    std::vector<float> result;
    result.reserve(values.size());
    for (const uint16_t &value : values) {
      result.push_back(fp16_ieee_to_fp32_value(value));
    }
    return result;
  }

  static float TruncToF16(float value) {
    return fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(value));
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<uint16_t> inputs_;
  std::vector<uint16_t> output_;
};

TEST_F(F16SimdSCALARTest, SetZero) {
  xnn_storeu_f16(output_.data(), xnn_zero_f16());
  EXPECT_THAT(ToFloat32(output_), testing::Each(testing::Eq(0.0f)));
}

TEST_F(F16SimdSCALARTest, Add) {
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

TEST_F(F16SimdSCALARTest, Mul) {
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

TEST_F(F16SimdSCALARTest, Fmadd) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t c =
      xnn_loadu_f16(inputs_.data() + 2 * xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_fmadd_f16(a, b, c);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k],
              TruncToF16(inputs_f32[k] * inputs_f32[k + xnn_simd_size_f16] +
                         inputs_f32[k + 2 * xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdSCALARTest, Fmsub) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t c =
      xnn_loadu_f16(inputs_.data() + 2 * xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_fmsub_f16(a, b, c);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k],
              TruncToF16(inputs_f32[k] * inputs_f32[k + xnn_simd_size_f16] -
                         inputs_f32[k + 2 * xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdSCALARTest, Fnmadd) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t c =
      xnn_loadu_f16(inputs_.data() + 2 * xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_fnmadd_f16(a, b, c);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k],
              TruncToF16(-inputs_f32[k] * inputs_f32[k + xnn_simd_size_f16] +
                         inputs_f32[k + 2 * xnn_simd_size_f16]));
  }
}

TEST_F(F16SimdSCALARTest, Sub) {
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

TEST_F(F16SimdSCALARTest, Div) {
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

TEST_F(F16SimdSCALARTest, Max) {
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

TEST_F(F16SimdSCALARTest, Min) {
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

TEST_F(F16SimdSCALARTest, Abs) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t res = xnn_abs_f16(a);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k], std::abs(inputs_f32[k]));
  }
}

TEST_F(F16SimdSCALARTest, Neg) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t res = xnn_neg_f16(a);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k], -inputs_f32[k]);
  }
}

TEST_F(F16SimdSCALARTest, And) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_and_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_[k], inputs_[k] & inputs_[k + xnn_simd_size_f16]);
  }
}

TEST_F(F16SimdSCALARTest, Or) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_or_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_[k], inputs_[k] | inputs_[k + xnn_simd_size_f16]);
  }
}

TEST_F(F16SimdSCALARTest, Xor) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t b = xnn_loadu_f16(inputs_.data() + xnn_simd_size_f16);
  const xnn_simd_f16_t res = xnn_xor_f16(a, b);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_[k], inputs_[k] ^ inputs_[k + xnn_simd_size_f16]);
  }
}

TEST_F(F16SimdSCALARTest, ShiftLeft) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  // Not using a loop since the `bits` parameter needs to be a compile-time
  // constant, e.g. for `neon`.
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 1);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 1) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 2);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 2) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 3);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 3) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 4);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 4) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 5);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 5) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 6);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 6) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 7);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 7) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 8);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 8) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 9);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 9) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 10);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 10) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 11);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 11) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 12);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 12) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 13);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 13) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 14);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 14) & 0xFFFF);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sll_f16(a, 15);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], (inputs_[k] << 15) & 0xFFFF);
    }
  }
}

TEST_F(F16SimdSCALARTest, ShiftRight) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  // Not using a loop since the `bits` parameter needs to be a compile-time
  // constant, e.g. for `neon`.
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 1);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 1);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 2);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 2);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 3);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 3);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 4);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 4);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 5);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 5);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 6);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 6);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 7);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 7);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 8);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 8);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 9);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 9);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 10);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 10);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 11);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 11);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 12);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 12);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 13);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 13);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 14);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 14);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_srl_f16(a, 15);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 15);
    }
  }
}

TEST_F(F16SimdSCALARTest, ShiftRightSigned) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  // Not using a loop since the `bits` parameter needs to be a compile-time
  // constant, e.g. for `neon`.
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 1);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 1);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 2);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 2);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 3);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 3);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 4);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 4);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 5);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 5);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 6);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 6);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 7);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 7);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 8);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 8);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 9);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 9);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 10);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 10);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 11);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 11);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 12);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 12);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 13);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 13);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 14);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 14);
    }
  }
  {
    const xnn_simd_f16_t res = xnn_sra_f16(a, 15);
    xnn_storeu_f16(output_.data(), res);
    for (size_t k = 0; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_[k], inputs_[k] >> 15);
    }
  }
}

TEST_F(F16SimdSCALARTest, CmpEq) {
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
    ASSERT_EQ(output_[k],
              inputs_[k] == inputs_[k + xnn_simd_size_f16] ? 0xFFFF : 0);
  }
}

TEST_F(F16SimdSCALARTest, GetExp) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  const xnn_simd_f16_t res = xnn_getexp_f16(a);
  xnn_storeu_f16(output_.data(), res);
  std::vector<float> output_f32 = ToFloat32(output_);
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t k = 0; k < xnn_simd_size_f16; k++) {
    ASSERT_EQ(output_f32[k], std::logb(inputs_f32[k]));
  }
}

TEST_F(F16SimdSCALARTest, StoreTail) {
  const xnn_simd_f16_t a = xnn_loadu_f16(inputs_.data());
  std::vector<float> inputs_f32 = ToFloat32(inputs_);
  for (size_t num_elements = 1; num_elements < xnn_simd_size_f16;
      num_elements++) {
    std::fill(output_.begin(), output_.end(), fp16_ieee_from_fp32_value(0.0f));
    xnn_store_tail_f16(output_.data(), a, num_elements);
    std::vector<float> output_f32 = ToFloat32(output_);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_f32[k], inputs_f32[k]);
    }
    for (size_t k = num_elements; k < xnn_simd_size_f16; k++) {
      ASSERT_EQ(output_f32[k], fp16_ieee_from_fp32_value(0.0f));
    }
  }
}

}  // namespace xnnpack

