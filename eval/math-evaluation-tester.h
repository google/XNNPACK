// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <ios>
#include <limits>
#include <vector>


#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


class MathEvaluationTester {
 public:
  inline MathEvaluationTester& input_value(float value) {
    this->input_min_ = value;
    this->input_max_ = value;
    return *this;
  }

  inline MathEvaluationTester& input_range(float lower_bound, float upper_bound) {
    this->input_min_ = lower_bound;
    this->input_max_ = upper_bound;
    return *this;
  }

  inline float input_min() const {
    return this->input_min_;
  }

  inline float input_max() const {
    return this->input_max_;
  }

  void TestOutputMatchReference(xnn_f32_unary_math_fn math_fn, float output_value) const {
    ASSERT_FALSE(std::isnan(output_value));
    ASSERT_FALSE(std::isnan(input_min()));
    ASSERT_FALSE(std::isnan(input_max()));
    ASSERT_LE(input_min(), input_max());
    ASSERT_EQ(std::signbit(input_min()), std::signbit(input_max()));

    uint32_t range_min = float_as_uint32(std::fabs(input_min()));
    uint32_t range_max = float_as_uint32(std::fabs(input_max()));
    uint32_t range_sign = std::signbit(input_min()) ? UINT32_C(0x80000000) : 0;
    if (range_min > range_max) {
      std::swap(range_min, range_max);
    }

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t block_start = range_min; block_start <= range_max; block_start += kBlockSize) {
      for (uint32_t block_offset = 0; block_offset < kBlockSize; block_offset++) {
        inputs[block_offset] = uint32_as_float(range_sign | std::min<uint32_t>(block_start + block_offset, range_max));
      }
      math_fn(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        const uint32_t reference_output = float_as_uint32(output_value);
        ASSERT_EQ(reference_output, float_as_uint32(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << " (" << inputs[i] << ")"
          << ", reference = 0x" << std::hex << std::setw(8) << std::setfill('0') << reference_output
          << " (" << output_value << ")"
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  void TestOutputMatchZero(xnn_f32_unary_math_fn math_fn) const {
    ASSERT_FALSE(std::isnan(input_min()));
    ASSERT_FALSE(std::isnan(input_max()));
    ASSERT_LE(input_min(), input_max());
    ASSERT_EQ(std::signbit(input_min()), std::signbit(input_max()));

    uint32_t range_min = float_as_uint32(std::fabs(input_min()));
    uint32_t range_max = float_as_uint32(std::fabs(input_max()));
    uint32_t range_sign = std::signbit(input_min()) ? UINT32_C(0x80000000) : 0;
    if (range_min > range_max) {
      std::swap(range_min, range_max);
    }

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);
    for (uint32_t block_start = range_min; block_start <= range_max; block_start += kBlockSize) {
      for (uint32_t block_offset = 0; block_offset < kBlockSize; block_offset++) {
        inputs[block_offset] = uint32_as_float(range_sign | std::min<uint32_t>(block_start + block_offset, range_max));
      }
      math_fn(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_EQ(0.0f, outputs[i])
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << " (" << inputs[i] << ")"
          << ", optimized = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

  void TestNaN(xnn_f32_unary_math_fn math_fn) const {
    ASSERT_TRUE(std::isnan(input_min()));
    ASSERT_TRUE(std::isnan(input_max()));

    const uint32_t range_min = UINT32_C(0x7F800001);
    const uint32_t range_max = UINT32_C(0x7FFFFFFF);

    std::vector<float, AlignedAllocator<float, 64>> inputs(kBlockSize);
    std::vector<float, AlignedAllocator<float, 64>> outputs(kBlockSize);

    // Positive NaN inputs
    for (uint32_t block_start = range_min; block_start <= range_max; block_start += kBlockSize) {
      for (uint32_t block_offset = 0; block_offset < kBlockSize; block_offset++) {
        inputs[block_offset] = uint32_as_float(std::min<uint32_t>(block_start + block_offset, range_max));
      }
      math_fn(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", output = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }

    // Negative NaN inputs
    for (uint32_t block_start = range_min; block_start <= range_max; block_start += kBlockSize) {
      for (uint32_t block_offset = 0; block_offset < kBlockSize; block_offset++) {
        inputs[block_offset] =
          uint32_as_float(UINT32_C(0x80000000) | std::min<uint32_t>(block_start + block_offset, range_max));
      }
      math_fn(kBlockSize * sizeof(float), inputs.data(), outputs.data());
      for (uint32_t i = 0; i < kBlockSize; i++) {
        ASSERT_TRUE(std::isnan(outputs[i]))
          << "input = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(inputs[i])
          << ", output = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(outputs[i]);
      }
    }
  }

 private:
  static constexpr int kBlockSize = 1024;

  float input_min_ = std::nanf("");
  float input_max_ = std::nanf("");
};
