// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "replicable_random_device.h"

class VHSwishMicrokernelTester {
 public:
  VHSwishMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  VHSwishMicrokernelTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  float input_scale() const {
    return this->input_scale_;
  }

  VHSwishMicrokernelTester& input_zero_point(int16_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  int16_t input_zero_point() const {
    return this->input_zero_point_;
  }

  VHSwishMicrokernelTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  float output_scale() const {
    return this->output_scale_;
  }

  VHSwishMicrokernelTester& output_zero_point(int16_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  int16_t output_zero_point() const {
    return this->output_zero_point_;
  }

  VHSwishMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_qs8_vhswish_ukernel_fn vhswish, xnn_init_qs8_hswish_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output(batch_size());
    std::vector<int8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));
      for (int i = 0; i < batch_size(); i++) {
        input[i] = i;
      }
      union xnn_qs8_hswish_params params;
      init_params(&params, input_zero_point(), output_zero_point(), input_scale(), output_scale());

      // Call optimized micro-kernel.
      vhswish(batch_size() * sizeof(int8_t), input.data(), output.data(), &params);

      // Compute reference results
      const int32_t input_scale_div = (int32_t) lrintf(256.0f * input_scale() / 6.0f);
      const int32_t scale_ratio = (int32_t) lrintf(256.0f * input_scale() / output_scale());
      for (size_t i = 0; i < batch_size(); i++) {
        const int32_t input_value = int32_t(uint32_t(input_zero_point() - input[i]) << 7);
        int32_t in = input_value * input_scale_div;
        in -= 16384;  // subtract 0.5 in Q15
        in = std::min<int32_t>(in, 0);
        in = std::max<int32_t>(in, -32768);
        const int32_t out = math_asr_s32(input_value * scale_ratio + INT32_C(0x4000), 15);
        int32_t output_value = math_asr_s32(in * out + INT32_C(0x4000), 15) + output_zero_point();
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

  void Test(xnn_qu8_vhswish_ukernel_fn vhswish, xnn_init_qu8_hswish_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(output_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<uint8_t>::max());

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<uint8_t> output(batch_size());
    std::vector<uint8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));
      for (int i = 0; i < batch_size(); i++) {
        input[i] = i;
      }
      union xnn_qu8_hswish_params params;
      init_params(&params, input_zero_point(), output_zero_point(), input_scale(), output_scale());

      // Call optimized micro-kernel.
      vhswish(batch_size() * sizeof(uint8_t), input.data(), output.data(), &params);

      // Compute reference results
      const int32_t input_scale_div = (int32_t) lrintf(256.0f * input_scale() / 6.0f);
      const int32_t scale_ratio = (int32_t) lrintf(256.0f * input_scale() / output_scale());
      for (size_t i = 0; i < batch_size(); i++) {
        const int32_t input_value = int32_t(uint32_t(input_zero_point() - input[i]) << 7);
        int32_t in = input_value * input_scale_div;
        in -= 16384;  // subtract 0.5 in Q15
        in = std::min<int32_t>(in, 0);
        in = std::max<int32_t>(in, -32768);
        const int32_t out = math_asr_s32(input_value * scale_ratio + INT32_C(0x4000), 15);
        int32_t output_value = math_asr_s32(in * out + INT32_C(0x4000), 15) + output_zero_point();
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
  float input_scale_ = 128.0f;
  float output_scale_ = 128.0f;
  int16_t input_zero_point_ = 1;
  int16_t output_zero_point_ = 5;
  size_t batch_size_ = 1;
  size_t iterations_ = 15;
};
