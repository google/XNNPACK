// clang-format off
// Auto-generated file. Do not edit!
//   Template: test/simd/u8-simd.cc.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// This header needs to go first for the arch test macros.
#include "src/xnnpack/common.h"

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

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
#include "src/xnnpack/simd/u8-wasmsimd.h"
#include "test/replicable_random_device.h"

namespace xnnpack {

class U8SimdWASMSIMDTest : public ::testing::Test {
 protected:
  void SetUp() override {
    inputs_.resize(3 * xnn_simd_size_u8);
    output_.resize(xnn_simd_size_u8);
    std::uniform_int_distribution<uint8_t> u8(-100, 100);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return u8(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<uint8_t> inputs_;
  std::vector<uint8_t> output_;
};

TEST_F(U8SimdWASMSIMDTest, Add) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  const xnn_simd_u8_t b = xnn_loadu_u8(inputs_.data() + xnn_simd_size_u8);
  const xnn_simd_u8_t res = xnn_add_u8(a, b);
  xnn_storeu_u8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_u8; k++) {
    ASSERT_EQ(output_[k], (uint8_t)(inputs_[k] + inputs_[k + xnn_simd_size_u8]));
  }
}

TEST_F(U8SimdWASMSIMDTest, Min) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  const xnn_simd_u8_t b = xnn_loadu_u8(inputs_.data() + xnn_simd_size_u8);
  const xnn_simd_u8_t res = xnn_min_u8(a, b);
  xnn_storeu_u8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_u8; k++) {
    ASSERT_EQ(output_[k], std::min<uint8_t>(inputs_[k], inputs_[k + xnn_simd_size_u8]));
  }
}

TEST_F(U8SimdWASMSIMDTest, Max) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  const xnn_simd_u8_t b = xnn_loadu_u8(inputs_.data() + xnn_simd_size_u8);
  const xnn_simd_u8_t res = xnn_max_u8(a, b);
  xnn_storeu_u8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_u8; k++) {
    ASSERT_EQ(output_[k], std::max<uint8_t>(inputs_[k], inputs_[k + xnn_simd_size_u8]));
  }
}

TEST_F(U8SimdWASMSIMDTest, Xor) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  const xnn_simd_u8_t b = xnn_loadu_u8(inputs_.data() + xnn_simd_size_u8);
  const xnn_simd_u8_t res = xnn_xor_u8(a, b);
  xnn_storeu_u8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_u8; k++) {
    ASSERT_EQ(output_[k], inputs_[k] ^ inputs_[k + xnn_simd_size_u8]);
  }
}

TEST_F(U8SimdWASMSIMDTest, StoreTail) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_u8;
      num_elements++) {
    xnn_store_tail_u8(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
    for (size_t k = num_elements; k < xnn_simd_size_u8; k++) {
      ASSERT_EQ(output_[k], 0.0f);
    }
  }
}
}  // namespace xnnpack

#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
