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


class SoftMaxOperatorTester {
 public:
  inline SoftMaxOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline SoftMaxOperatorTester& input_stride(size_t input_stride) {
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

  inline SoftMaxOperatorTester& output_stride(size_t output_stride) {
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

  inline SoftMaxOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline SoftMaxOperatorTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  inline float input_scale() const {
    return this->input_scale_;
  }

  inline SoftMaxOperatorTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline float output_scale() const {
    return 1.0f / 256.0f;
  }

  inline uint8_t output_zero_point() const {
    return 0;
  }

  inline SoftMaxOperatorTester& iterations(size_t iterations) {
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
        const int32_t max_input = *std::max_element(
          input.data() + i * input_stride(),
          input.data() + i * input_stride() + channels());
        float sum_exp = 0.0f;
        for (size_t c = 0; c < channels(); c++) {
          sum_exp +=
              std::exp((int32_t(input[i * input_stride() + c]) - max_input) *
                       input_scale());
        }
        for (size_t c = 0; c < channels(); c++) {
          output_ref[i * channels() + c] =
              std::exp((int32_t(input[i * input_stride() + c]) - max_input) *
                       input_scale()) /
              (sum_exp * output_scale());
          output_ref[i * channels() + c] = std::min(output_ref[i * channels() + c], 255.0f);
        }
      }

      // Create, setup, run, and destroy SoftMax operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t softmax_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_softmax_nc_q8(
          channels(), input_stride(), output_stride(),
          input_scale(),
          output_zero_point(), output_scale(),
          0, &softmax_op));
      ASSERT_NE(nullptr, softmax_op);

      // Smart pointer to automatically delete softmax_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_softmax_op(softmax_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_softmax_nc_q8(
          softmax_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(softmax_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(float(int32_t(output[i * output_stride() + c])), output_ref[i * channels() + c], 0.6f);
        }
      }
    }
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> input((batch_size() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    std::vector<double> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        const double max_input = *std::max_element(
          input.data() + i * input_stride(),
          input.data() + i * input_stride() + channels());
        double sum_exp = 0.0;
        for (size_t c = 0; c < channels(); c++) {
          sum_exp += std::exp(double(input[i * input_stride() + c]) - max_input);
        }
        for (size_t c = 0; c < channels(); c++) {
          output_ref[i * channels() + c] =
              std::exp(double(input[i * input_stride() + c]) - max_input) / sum_exp;
        }
      }

      // Create, setup, run, and destroy SoftMax operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t softmax_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_softmax_nc_f32(
          channels(), input_stride(), output_stride(),
          0, &softmax_op));
      ASSERT_NE(nullptr, softmax_op);

      // Smart pointer to automatically delete softmax_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_softmax_op(softmax_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_softmax_nc_f32(
          softmax_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(softmax_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
            double(output[i * output_stride() + c]),
            output_ref[i * channels() + c],
            output_ref[i * channels() + c] * 1.0e-4);
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  float input_scale_{0.176080093};
  uint8_t input_zero_point_{121};
  size_t iterations_{15};
};
