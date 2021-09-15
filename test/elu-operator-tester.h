// Copyright 2020 Google LLC
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
#include <random>
#include <vector>

#include <xnnpack.h>


class ELUOperatorTester {
 public:
  inline ELUOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline ELUOperatorTester& input_stride(size_t input_stride) {
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

  inline ELUOperatorTester& output_stride(size_t output_stride) {
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

  inline ELUOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline ELUOperatorTester& alpha(float alpha) {
    assert(alpha > 0.0f);
    assert(alpha < 1.0f);
    this->alpha_ = alpha;
    return *this;
  }

  inline float alpha() const {
    return this->alpha_;
  }

  inline ELUOperatorTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  inline float input_scale() const {
    return this->input_scale_;
  }

  inline ELUOperatorTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline ELUOperatorTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  inline float output_scale() const {
    return this->output_scale_;
  }

  inline ELUOperatorTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline ELUOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline ELUOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline ELUOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-20.0f, 20.0f), std::ref(rng));

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + (batch_size() - 1) * input_stride() + channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    std::vector<double> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const double x = double(input[i * input_stride() + c]);
          output_ref[i * channels() + c] = std::signbit(x) ? std::expm1(x) * alpha() : x;
        }
      }

      // Create, setup, run, and destroy ELU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t elu_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_elu_nc_f32(
          channels(), input_stride(), output_stride(),
          alpha(),
          0, &elu_op));
      ASSERT_NE(nullptr, elu_op);

      // Smart pointer to automatically delete elu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_elu_op(elu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_elu_nc_f32(
          elu_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(elu_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(output[i * output_stride() + c],
                      output_ref[i * channels() + c],
                      std::abs(output_ref[i * channels() + c]) * 1.0e-5)
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels()
            << ", input " << input[i * input_stride() + c] << ", alpha " << alpha();
        }
      }
    }
  }

  void TestQS8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i8rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
      std::ref(rng));

    std::vector<int8_t> input((batch_size() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(i8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = input_scale() *
            (int32_t(input[i * input_stride() + c]) - int32_t(input_zero_point() - 0x80));
          const float elu_x = std::signbit(x) ? alpha() * std::expm1(x) : x;
          const float scaled_elu_x = elu_x / output_scale();
          float y = scaled_elu_x;
          y = std::min<float>(y, int32_t(qmax() - 0x80) - int32_t(output_zero_point() - 0x80));
          y = std::max<float>(y, int32_t(qmin() - 0x80) - int32_t(output_zero_point() - 0x80));
          output_ref[i * channels() + c] = y + int32_t(output_zero_point() - 0x80);
        }
      }

      // Create, setup, run, and destroy Sigmoid operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t elu_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_elu_nc_qs8(
          channels(), input_stride(), output_stride(),
          alpha(),
          int8_t(input_zero_point() - 0x80), input_scale(),
          int8_t(output_zero_point() - 0x80), output_scale(),
          int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          0, &elu_op));
      ASSERT_NE(nullptr, elu_op);

      // Smart pointer to automatically delete elu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_sigmoid_op(elu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_elu_nc_qs8(
          elu_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(elu_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(float(int32_t(output[i * output_stride() + c])), output_ref[i * channels() + c], 0.6f);
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  float alpha_{0.5f};
  float input_scale_{0.75f};
  uint8_t input_zero_point_{121};
  float output_scale_{0.75f};
  uint8_t output_zero_point_{121};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
