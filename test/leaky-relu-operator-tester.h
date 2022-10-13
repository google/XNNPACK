// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
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
#include <limits>
#include <random>
#include <vector>

#include <fp16.h>

#include <xnnpack.h>

static uint16_t flush_fp16_denormal_to_zero(uint16_t v) {
  return (v & UINT16_C(0x7C00)) == 0 ? v & UINT16_C(0x8000) : v;
};


class LeakyReLUOperatorTester {
 public:
  inline LeakyReLUOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline LeakyReLUOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->input_stride_ >= this->channels_);
      return this->input_stride_;
    }
  }

  inline LeakyReLUOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->output_stride_ >= this->channels_);
      return this->output_stride_;
    }
  }

  inline LeakyReLUOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline LeakyReLUOperatorTester& negative_slope(float negative_slope) {
    assert(std::isnormal(negative_slope));
    this->negative_slope_ = negative_slope;
    return *this;
  }

  inline float negative_slope() const {
    return this->negative_slope_;
  }

  inline LeakyReLUOperatorTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  inline float input_scale() const {
    return this->input_scale_;
  }

  inline LeakyReLUOperatorTester& input_zero_point(int16_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline int16_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline LeakyReLUOperatorTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  inline float output_scale() const {
    return this->output_scale_;
  }

  inline LeakyReLUOperatorTester& output_zero_point(int16_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int16_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline LeakyReLUOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF16() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) + (batch_size() - 1) * input_stride() + channels());
    std::vector<uint16_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() {
        return flush_fp16_denormal_to_zero(fp16_ieee_from_fp32_value(f32dist(rng)));
      });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);
      const uint16_t negative_slope_as_half = fp16_ieee_from_fp32_value(negative_slope());
      const float negative_slope_as_float = fp16_ieee_to_fp32_value(negative_slope_as_half);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = fp16_ieee_to_fp32_value(input[i * input_stride() + c]);
          const float y = std::signbit(x) ? x * negative_slope_as_float : x;
          output_ref[i * channels() + c] = y;
        }
      }

      // Create, setup, run, and destroy Leaky ReLU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t leaky_relu_op = nullptr;

      const xnn_status status = xnn_create_leaky_relu_nc_f16(
          channels(), input_stride(), output_stride(),
          negative_slope(),
          0, &leaky_relu_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, leaky_relu_op);

      // Smart pointer to automatically delete leaky_relu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_leaky_relu_op(leaky_relu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_leaky_relu_nc_f16(
          leaky_relu_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(leaky_relu_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              fp16_ieee_to_fp32_value(output[i * output_stride() + c]),
              output_ref[i * channels() + c],
              std::max(2.0e-4f, std::abs(output_ref[i * channels() + c]) * 1.0e-3f))
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + (batch_size() - 1) * input_stride() + channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = input[i * input_stride() + c];
          const float y = std::signbit(x) ? x * negative_slope() : x;
          output_ref[i * channels() + c] = y;
        }
      }

      // Create, setup, run, and destroy Leaky ReLU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t leaky_relu_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_leaky_relu_nc_f32(
          channels(), input_stride(), output_stride(),
          negative_slope(),
          0, &leaky_relu_op));
      ASSERT_NE(nullptr, leaky_relu_op);

      // Smart pointer to automatically delete leaky_relu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_leaky_relu_op(leaky_relu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_leaky_relu_nc_f32(
          leaky_relu_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(leaky_relu_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_EQ(output[i * output_stride() + c], output_ref[i * channels() + c])
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels()
            << ", input " << input[i * input_stride() + c] << ", negative slope " << negative_slope();
        }
      }
    }
  }

  void TestQS8() const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) + (batch_size() - 1) * input_stride() + channels());
    std::vector<int8_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = input_scale() * (int32_t(input[i * input_stride() + c]) - input_zero_point());
          float y = (x < 0.0f ? x * negative_slope() : x) / output_scale() + float(output_zero_point());
          y = std::max<float>(y, std::numeric_limits<int8_t>::min());
          y = std::min<float>(y, std::numeric_limits<int8_t>::max());
          output_ref[i * channels() + c] = y;
        }
      }

      // Create, setup, run, and destroy Leaky ReLU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t leaky_relu_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_leaky_relu_nc_qs8(
          channels(), input_stride(), output_stride(),
          negative_slope(),
          input_zero_point(), input_scale(),
          output_zero_point(), output_scale(),
          0, &leaky_relu_op));
      ASSERT_NE(nullptr, leaky_relu_op);

      // Smart pointer to automatically delete leaky_relu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_leaky_relu_op(leaky_relu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_leaky_relu_nc_qs8(
          leaky_relu_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(leaky_relu_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(float(int32_t(output[i * output_stride() + c])), output_ref[i * channels() + c], 0.9f)
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels()
            << ", input " << int32_t(input[i * input_stride() + c])
            << ", input zero point " << input_zero_point() << ", output zero point " << output_zero_point()
            << ", positive input-to-output ratio " << (input_scale() / output_scale())
            << ", negative input-to-output ratio " << (input_scale() / output_scale() * negative_slope());
        }
      }
    }
  }

  void TestQU8() const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(output_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<uint8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + (batch_size() - 1) * input_stride() + channels());
    std::vector<uint8_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = input_scale() * (int32_t(input[i * input_stride() + c]) - input_zero_point());
          float y = (x < 0.0f ? x * negative_slope() : x) / output_scale() + float(output_zero_point());
          y = std::max<float>(y, std::numeric_limits<uint8_t>::min());
          y = std::min<float>(y, std::numeric_limits<uint8_t>::max());
          output_ref[i * channels() + c] = y;
        }
      }

      // Create, setup, run, and destroy Leaky ReLU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t leaky_relu_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_leaky_relu_nc_qu8(
          channels(), input_stride(), output_stride(),
          negative_slope(),
          input_zero_point(), input_scale(),
          output_zero_point(), output_scale(),
          0, &leaky_relu_op));
      ASSERT_NE(nullptr, leaky_relu_op);

      // Smart pointer to automatically delete leaky_relu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_leaky_relu_op(leaky_relu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_leaky_relu_nc_qu8(
          leaky_relu_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(leaky_relu_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(float(int32_t(output[i * output_stride() + c])), output_ref[i * channels() + c], 0.9f)
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels()
            << ", input " << int32_t(input[i * input_stride() + c])
            << ", input zero point " << input_zero_point() << ", output zero point " << output_zero_point()
            << ", positive input-to-output ratio " << (input_scale() / output_scale())
            << ", negative input-to-output ratio " << (input_scale() / output_scale() * negative_slope());
        }
      }
    }
  }

  void TestRunF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + (batch_size() - 1) * input_stride() + channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = input[i * input_stride() + c];
          const float y = std::signbit(x) ? x * negative_slope() : x;
          output_ref[i * channels() + c] = y;
        }
      }

      // Create, setup, run, and destroy Leaky ReLU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_leaky_relu_nc_f32(
          channels(),
          input_stride(),
          output_stride(),
          batch_size(),
          input.data(),
          output.data(),
          negative_slope(),
          0,
          nullptr  /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_EQ(output[i * output_stride() + c], output_ref[i * channels() + c])
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels()
            << ", input " << input[i * input_stride() + c] << ", negative slope " << negative_slope();
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  float negative_slope_{0.3f};
  float output_scale_{0.75f};
  int16_t output_zero_point_{53};
  float input_scale_{1.25f};
  int16_t input_zero_point_{41};
  size_t iterations_{15};
};
