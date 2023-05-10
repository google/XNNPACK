// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ios>
#include <limits>
#include <utility>
#include <vector>

#include <fp16/fp16.h>

#include "math-evaluation-tester.h"

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void MathEvaluationTester::TestOutputMatchReference(xnn_f16_unary_math_fn math_fn, float output_value) const {
  ASSERT_FALSE(std::isnan(output_value));
  ASSERT_FALSE(std::isnan(input_min()));
  ASSERT_FALSE(std::isnan(input_max()));
  ASSERT_LE(input_min(), input_max());
  ASSERT_EQ(std::signbit(input_min()), std::signbit(input_max()));
  ASSERT_EQ(fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_value)), output_value);
  ASSERT_EQ(fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(input_min())), input_min());
  ASSERT_EQ(fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(input_max())), input_max());

  uint16_t range_min = fp16_ieee_from_fp32_value(std::fabs(input_min()));
  uint16_t range_max = fp16_ieee_from_fp32_value(std::fabs(input_max()));
  uint16_t range_sign = std::signbit(input_min()) ? UINT16_C(0x8000) : 0;
  if (range_min > range_max) {
    std::swap(range_min, range_max);
  }

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);
  for (uint16_t block_start = range_min; block_start <= range_max; block_start += kBlockSize) {
    for (uint16_t block_offset = 0; block_offset < kBlockSize; block_offset++) {
      inputs[block_offset] = range_sign | std::min<uint16_t>(block_start + block_offset, range_max);
    }
    math_fn(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint16_t i = 0; i < kBlockSize; i++) {
      const uint16_t reference_output = fp16_ieee_from_fp32_value(output_value);
      ASSERT_EQ(reference_output, outputs[i])
        << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
        << " (" << fp16_ieee_to_fp32_value(inputs[i]) << ")"
        << ", reference = 0x" << std::hex << std::setw(4) << std::setfill('0') << reference_output
        << " (" << output_value << ")"
        << ", optimized = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i]
        << " (" << fp16_ieee_to_fp32_value(outputs[i]) << ")";
    }
  }
}

void MathEvaluationTester::TestOutputMatchReference(xnn_f32_unary_math_fn math_fn, float output_value) const {
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

void MathEvaluationTester::TestOutputMatchZero(xnn_f32_unary_math_fn math_fn) const {
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

void MathEvaluationTester::TestNaN(xnn_f16_unary_math_fn math_fn) const {
  ASSERT_TRUE(std::isnan(input_min()));
  ASSERT_TRUE(std::isnan(input_max()));

  const uint16_t range_min = UINT16_C(0x7C01);
  const uint16_t range_max = UINT16_C(0x7FFF);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> inputs(kBlockSize);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> outputs(kBlockSize);

  // Positive NaN inputs
  for (uint16_t block_start = range_min; block_start <= range_max; block_start += kBlockSize) {
    for (uint16_t block_offset = 0; block_offset < kBlockSize; block_offset++) {
      inputs[block_offset] = std::min<uint16_t>(block_start + block_offset, range_max);
    }
    math_fn(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint16_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(fp16_ieee_to_fp32_value(outputs[i])))
        << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
        << ", output = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }

  // Negative NaN inputs
  for (uint16_t block_start = range_min; block_start <= range_max; block_start += kBlockSize) {
    for (uint16_t block_offset = 0; block_offset < kBlockSize; block_offset++) {
      inputs[block_offset] =
        UINT16_C(0x8000) | std::min<uint16_t>(block_start + block_offset, range_max);
    }
    math_fn(kBlockSize * sizeof(uint16_t), inputs.data(), outputs.data());
    for (uint16_t i = 0; i < kBlockSize; i++) {
      ASSERT_TRUE(std::isnan(fp16_ieee_to_fp32_value(outputs[i])))
        << "input = 0x" << std::hex << std::setw(4) << std::setfill('0') << inputs[i]
        << ", output = 0x" << std::hex << std::setw(4) << std::setfill('0') << outputs[i];
    }
  }
}

void MathEvaluationTester::TestNaN(xnn_f32_unary_math_fn math_fn) const {
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
