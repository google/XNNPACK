// Auto-generated file. Do not edit!
//   Template: test/s8-simd.cc.in
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
#include <limits>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xnnpack/isa-checks.h"
#include "xnnpack/simd/s8-scalar.h"
#include "replicable_random_device.h"

namespace xnnpack {

class S8SimdSCALARTest : public ::testing::Test {
 protected:
  void SetUp() override {
    inputs_.resize(3 * xnn_simd_size_s8);
    output_.resize(xnn_simd_size_s8);
    std::uniform_int_distribution<int8_t> s8(-100, 100);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return s8(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<int8_t> inputs_;
  std::vector<int8_t> output_;
};

TEST_F(S8SimdSCALARTest, Add) {
  const xnn_simd_s8_t a = xnn_loadu_s8(inputs_.data());
  const xnn_simd_s8_t b = xnn_loadu_s8(inputs_.data() + xnn_simd_size_s8);
  const xnn_simd_s8_t res = xnn_add_s8(a, b);
  xnn_storeu_s8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s8; k++) {
    ASSERT_EQ(output_[k], (int8_t)(inputs_[k] + inputs_[k + xnn_simd_size_s8]));
  }
}

TEST_F(S8SimdSCALARTest, StoreTail) {
  const xnn_simd_s8_t a = xnn_loadu_s8(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s8;
      num_elements++) {
    std::fill(output_.begin(), output_.end(), 0.0f);
    xnn_store_tail_s8(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
    for (size_t k = num_elements; k < xnn_simd_size_s8; k++) {
      ASSERT_EQ(output_[k], 0.0f);
    }
  }
}
}  // namespace xnnpack

