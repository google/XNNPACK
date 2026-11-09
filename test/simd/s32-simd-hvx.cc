// clang-format off
// Auto-generated file. Do not edit!
//   Template: test/simd/s32-simd.cc.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// This header needs to go first for the arch test macros.
#include "src/xnnpack/common.h"

#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "src/xnnpack/isa-checks.h"


#include "src/xnnpack/simd/s32-hvx.h"


#include "test/replicable_random_device.h"

namespace xnnpack {

class S32SimdHVXTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_hvx);
    inputs_.resize(3 * xnn_simd_size_s32);
    output_.resize(xnn_simd_size_s32);
    std::uniform_int_distribution<int> s32dist(-10000, 10000);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return s32dist(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<int32_t> inputs_;
  std::vector<int32_t> output_;
};


TEST_F(S32SimdHVXTest, Mul) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_mul_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] * inputs_[k + xnn_simd_size_s32]);
  }
}

TEST_F(S32SimdHVXTest, Sub) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_sub_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] - inputs_[k + xnn_simd_size_s32]);
  }
}

TEST_F(S32SimdHVXTest, Min) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_min_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], std::min(inputs_[k], inputs_[k + xnn_simd_size_s32]));
  }
}

TEST_F(S32SimdHVXTest, Max) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_max_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], std::max(inputs_[k], inputs_[k + xnn_simd_size_s32]));
  }
}

TEST_F(S32SimdHVXTest, StoreTail) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s32;
      num_elements++) {
    xnn_store_tail_s32(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]);
    }
    for (size_t k = num_elements; k < xnn_simd_size_s32; k++) {
      ASSERT_EQ(output_[k], 0.0f);
    }
  }
}

TEST_F(S32SimdHVXTest, CvtF32S32) {
  std::vector<int32_t> values = {
      0, 1, -1, 2, -2, 255, -256, 65535, 65536, -65537,
      (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + 3, -(1 << 24) - 1,
      (1 << 25) + 2, (1 << 25) + 6, 0x7FFFFF80, 0x7FFFFFC0,
      std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::min() + 1};
  std::uniform_int_distribution<int32_t> s32dist(
      std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
  while (values.size() % xnn_simd_size_s32 != 0 || values.size() < 1024) {
    values.push_back(s32dist(rng_));
  }
  std::vector<float> result(xnn_simd_size_s32);
  for (size_t i = 0; i < values.size(); i += xnn_simd_size_s32) {
    const xnn_simd_s32_t a = xnn_loadu_s32(values.data() + i);
    const HVX_Vector res = xnn_cvt_f32_s32(a);
    std::memcpy(result.data(), &res, sizeof(res));
    for (size_t k = 0; k < xnn_simd_size_s32; k++) {
      ASSERT_EQ(result[k], static_cast<float>(values[i + k]))
          << "input " << values[i + k];
    }
  }
}
}  // namespace xnnpack

#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
