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
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>


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
    assert(negative_slope > 0.0f);
    assert(negative_slope < 1.0f);
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

  inline LeakyReLUOperatorTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline uint8_t input_zero_point() const {
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

  inline LeakyReLUOperatorTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline LeakyReLUOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline LeakyReLUOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline LeakyReLUOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestQ8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input((batch_size() - 1) * input_stride() + channels());
    std::vector<uint8_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = input_scale() * (int32_t(input[i * input_stride() + c]) - int32_t(input_zero_point()));
          float y = (x < 0.0f ? x * negative_slope() : x) / output_scale();
          y = std::min<float>(y, int32_t(qmax()) - int32_t(output_zero_point()));
          y = std::max<float>(y, int32_t(qmin()) - int32_t(output_zero_point()));
          output_ref[i * channels() + c] = y + float(int32_t(output_zero_point()));
        }
      }

      // Create, setup, run, and destroy LeakyReLU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t leaky_relu_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_leaky_relu_nc_q8(
          channels(), input_stride(), output_stride(),
          negative_slope(),
          input_zero_point(), input_scale(),
          output_zero_point(), output_scale(),
          qmin(), qmax(),
          0, &leaky_relu_op));
      ASSERT_NE(nullptr, leaky_relu_op);

      // Smart pointer to automatically delete leaky_relu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_leaky_relu_op(leaky_relu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_leaky_relu_nc_q8(
          leaky_relu_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(leaky_relu_op, nullptr /* thread pool */));

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
  float negative_slope_{0.5f};
  float output_scale_{0.75f};
  uint8_t output_zero_point_{133};
  float input_scale_{1.25f};
  uint8_t input_zero_point_{121};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
