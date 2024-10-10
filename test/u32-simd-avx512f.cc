// Auto-generated file. Do not edit!
//   Template: test/u32-simd.cc.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// This header needs to go first for the arch test macros.
#include "xnnpack/common.h"

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
#include "xnnpack/isa-checks.h"
#include "xnnpack/simd/u32-avx512f.h"
#include "replicable_random_device.h"

namespace xnnpack {

class U32SimdAVX512FTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TEST_REQUIRES_X86_AVX512F;
    inputs_.resize(3 * xnn_simd_size_u32);
    output_.resize(xnn_simd_size_u32);
    std::uniform_int_distribution<uint32_t> u32dist(0, 20000);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return u32dist(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<uint32_t> inputs_;
  std::vector<uint32_t> output_;
};

TEST_F(U32SimdAVX512FTest, Mul) {
  const xnn_simd_u32_t a = xnn_loadu_u32(inputs_.data());
  const xnn_simd_u32_t b = xnn_loadu_u32(inputs_.data() + xnn_simd_size_u32);
  const xnn_simd_u32_t res = xnn_mul_u32(a, b);
  xnn_storeu_u32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_u32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] * inputs_[k + xnn_simd_size_u32]);
  }
}

TEST_F(U32SimdAVX512FTest, StoreTail) {
  const xnn_simd_u32_t a = xnn_loadu_u32(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_u32;
      num_elements++) {
    xnn_store_tail_u32(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]);
    }
    for (size_t k = num_elements; k < xnn_simd_size_u32; k++) {
      ASSERT_EQ(output_[k], 0.0f);
    }
  }
}
}  // namespace xnnpack

#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
