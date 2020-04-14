// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>


class GlobalAveragePoolingOperatorTester {
 public:
  inline GlobalAveragePoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline GlobalAveragePoolingOperatorTester& width(size_t width) {
    assert(width != 0);
    this->width_ = width;
    return *this;
  }

  inline size_t width() const {
    return this->width_;
  }

  inline GlobalAveragePoolingOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return channels();
    } else {
      assert(this->input_stride_ >= channels());
      return this->input_stride_;
    }
  }

  inline GlobalAveragePoolingOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  inline GlobalAveragePoolingOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline GlobalAveragePoolingOperatorTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  inline float input_scale() const {
    return this->input_scale_;
  }

  inline GlobalAveragePoolingOperatorTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline GlobalAveragePoolingOperatorTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  inline float output_scale() const {
    return this->output_scale_;
  }

  inline GlobalAveragePoolingOperatorTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline GlobalAveragePoolingOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GlobalAveragePoolingOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GlobalAveragePoolingOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestNWCxQ8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input((batch_size() * width() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output(batch_size() * output_stride());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      const double scale = double(input_scale()) / (double(width()) * double(output_scale()));
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          double acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += double(int32_t(input[(i * width() + k) * input_stride() + j]) - int32_t(input_zero_point()));
          }
          output_ref[i * channels() + j] = float(acc * scale + double(output_zero_point()));
          output_ref[i * channels() + j] = std::min<float>(output_ref[i * channels() + j], float(qmax()));
          output_ref[i * channels() + j] = std::max<float>(output_ref[i * channels() + j], float(qmin()));
        }
      }

      // Create, setup, run, and destroy Global Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_average_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_global_average_pooling_nwc_q8(
          channels(), input_stride(), output_stride(),
          input_zero_point(), input_scale(),
          output_zero_point(), output_scale(),
          qmin(), qmax(),
          0, &global_average_pooling_op));
      ASSERT_NE(nullptr, global_average_pooling_op);

      // Smart pointer to automatically delete global_average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_average_pooling_op(global_average_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_average_pooling_nwc_q8(
          global_average_pooling_op,
          batch_size(), width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_average_pooling_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(uint32_t(output[i * output_stride() + c]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(output[i * output_stride() + c]), uint32_t(qmin()));
          ASSERT_NEAR(float(int32_t(output[i * output_stride() + c])), output_ref[i * channels() + c], 0.80f) <<
            "in batch index " << i << ", channel " << c;
        }
      }
    }
  }

  void TestNWCxF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> input((batch_size() * width() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output(batch_size() * output_stride());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          float acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += input[(i * width() + k) * input_stride() + j];
          }
          output_ref[i * channels() + j] = acc / float(width());
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_range == 0.0f ?
        -std::numeric_limits<float>::infinity() :
        accumulated_min + accumulated_range / 255.0f * float(qmin());
      const float output_max = accumulated_range == 0.0f ?
        +std::numeric_limits<float>::infinity() :
        accumulated_max - accumulated_range / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Global Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_average_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_global_average_pooling_nwc_f32(
          channels(), input_stride(), output_stride(),
          output_min, output_max,
          0, &global_average_pooling_op));
      ASSERT_NE(nullptr, global_average_pooling_op);

      // Smart pointer to automatically delete global_average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_average_pooling_op(global_average_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_average_pooling_nwc_f32(
          global_average_pooling_op,
          batch_size(), width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_average_pooling_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(output[i * output_stride() + c], output_max);
          ASSERT_GE(output[i * output_stride() + c], output_min);
          ASSERT_NEAR(output[i * output_stride() + c], output_ref[i * channels() + c], std::abs(output_ref[i * channels() + c]) * 1.0e-6f) <<
            "in batch index " << i << ", channel " << c;
        }
      }
    }
  }

  void TestNCWxF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> input(batch_size() * channels() * width() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output(batch_size() * channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          float acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += input[(i * channels() + j) * width() + k];
          }
          output_ref[i * channels() + j] = acc / float(width());
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_range == 0.0f ?
        -std::numeric_limits<float>::infinity() :
        accumulated_min + accumulated_range / 255.0f * float(qmin());
      const float output_max = accumulated_range == 0.0f ?
        +std::numeric_limits<float>::infinity() :
        accumulated_max - accumulated_range / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Global Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_average_pooling_op = nullptr;

      xnn_status status = xnn_create_global_average_pooling_ncw_f32(
        channels(), output_min, output_max,
        0, &global_average_pooling_op);
      if (status == xnn_status_unsupported_parameter) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);

      // Smart pointer to automatically delete global_average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_average_pooling_op(global_average_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_average_pooling_ncw_f32(
          global_average_pooling_op,
          batch_size(), width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_average_pooling_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(output[i * channels() + c], output_max);
          ASSERT_GE(output[i * channels() + c], output_min);
          ASSERT_NEAR(output[i * channels() + c], output_ref[i * channels() + c], std::abs(output_ref[i * channels() + c]) * 1.0e-5f) <<
            "in batch index " << i << ", channel " << c;
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t width_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  float input_scale_{1.0f};
  float output_scale_{1.0f};
  uint8_t input_zero_point_{121};
  uint8_t output_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
