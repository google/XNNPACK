// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/math.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


class VLReLUMicrokernelTester {
 public:
  inline VLReLUMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline VLReLUMicrokernelTester& positive_scale(float positive_scale) {
    assert(positive_scale > 0.0f);
    assert(std::isnormal(positive_scale));
    this->positive_scale_ = positive_scale;
    return *this;
  }

  inline float positive_scale() const {
    return this->positive_scale_;
  }

  inline VLReLUMicrokernelTester& negative_scale(float negative_scale) {
    assert(std::isnormal(negative_scale));
    this->negative_scale_ = negative_scale;
    return *this;
  }

  inline float negative_scale() const {
    return this->negative_scale_;
  }

  inline VLReLUMicrokernelTester& input_zero_point(int16_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline int16_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline VLReLUMicrokernelTester& output_zero_point(int16_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int16_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline VLReLUMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_qs8_vlrelu_ukernel_fn vlrelu, xnn_init_qs8_lrelu_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output(batch_size());
    std::vector<int8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      union xnn_qs8_lrelu_params params;
      init_params(&params, positive_scale(), negative_scale(), input_zero_point(), output_zero_point());

      // Call optimized micro-kernel.
      vlrelu(batch_size() * sizeof(int8_t), input.data(), output.data(), &params);

      // Compute reference results
      const int32_t positive_multiplier = (int32_t) lrintf(-256.0f * positive_scale());
      const int32_t negative_multiplier = (int32_t) lrintf(-256.0f * negative_scale());
      for (size_t i = 0; i < batch_size(); i++) {
        const int32_t input_value = (input_zero_point() - input[i]) * 128;
        const int32_t multiplier = input_value <= 0 ? positive_multiplier : negative_multiplier;
        int32_t output_value = math_asr_s32(input_value * multiplier + INT32_C(0x4000), 15) + output_zero_point();
        output_value = std::min<int32_t>(output_value, std::numeric_limits<int8_t>::max());
        output_value = std::max<int32_t>(output_value, std::numeric_limits<int8_t>::min());
        output_ref[i] = static_cast<int8_t>(output_value);
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

  void Test(xnn_qu8_vlrelu_ukernel_fn vlrelu, xnn_init_qu8_lrelu_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(output_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<uint8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output(batch_size());
    std::vector<uint8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      union xnn_qu8_lrelu_params params;
      init_params(&params, positive_scale(), negative_scale(), input_zero_point(), output_zero_point());

      // Call optimized micro-kernel.
      vlrelu(batch_size() * sizeof(uint8_t), input.data(), output.data(), &params);

      // Compute reference results
      const int32_t positive_multiplier = (int32_t) lrintf(-256.0f * positive_scale());
      const int32_t negative_multiplier = (int32_t) lrintf(-256.0f * negative_scale());
      for (size_t i = 0; i < batch_size(); i++) {
        const int32_t input_value = (input_zero_point() - input[i]) * 128;
        const int32_t multiplier = input_value <= 0 ? positive_multiplier : negative_multiplier;
        int32_t output_value = math_asr_s32(input_value * multiplier + INT32_C(0x4000), 15) + output_zero_point();
        output_value = std::min<int32_t>(output_value, std::numeric_limits<uint8_t>::max());
        output_value = std::max<int32_t>(output_value, std::numeric_limits<uint8_t>::min());
        output_ref[i] = static_cast<uint8_t>(output_value);
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

 private:
  float positive_scale_ = 1.75f;
  float negative_scale_ = 0.75f;
  int16_t input_zero_point_ = 1;
  int16_t output_zero_point_ = 5;
  size_t batch_size_ = 1;
  size_t iterations_ = 15;
};
