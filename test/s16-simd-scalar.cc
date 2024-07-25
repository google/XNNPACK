// Auto-generated file. Do not edit!
//   Template: test/s16-simd.cc.in
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
#include "xnnpack/simd/s16-scalar.h"
#include "replicable_random_device.h"

namespace xnnpack {

class S16SimdSCALARTest : public ::testing::Test {
 protected:
  void SetUp() override {
    inputs_.resize(3 * xnn_simd_size_s16);
    output_.resize(xnn_simd_size_s16);
    std::uniform_int_distribution<int16_t> s16(-100, 100);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return s16(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<int16_t> inputs_;
  std::vector<int16_t> output_;
};

TEST_F(S16SimdSCALARTest, StoreTail) {
  const xnn_simd_s16_t a = xnn_loadu_s16(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s16;
      num_elements++) {
    std::fill(output_.begin(), output_.end(), 0.0f);
    xnn_store_tail_s16(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
    for (size_t k = num_elements; k < xnn_simd_size_s16; k++) {
      ASSERT_EQ(output_[k], 0.0f);
    }
  }
}
}  // namespace xnnpack

