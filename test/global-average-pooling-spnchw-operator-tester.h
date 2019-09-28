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
#include <random>
#include <vector>

#include <xnnpack.h>


class GlobalAveragePoolingSpNCHWOperatorTester {
 public:
  inline GlobalAveragePoolingSpNCHWOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline GlobalAveragePoolingSpNCHWOperatorTester& height(size_t height) {
    assert(height != 0);
    this->height_ = height;
    return *this;
  }

  inline size_t height() const {
    return this->height_;
  }

  inline GlobalAveragePoolingSpNCHWOperatorTester& width(size_t width) {
    assert(width != 0);
    this->width_ = width;
    return *this;
  }

  inline size_t width() const {
    return this->width_;
  }

  inline GlobalAveragePoolingSpNCHWOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline GlobalAveragePoolingSpNCHWOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GlobalAveragePoolingSpNCHWOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GlobalAveragePoolingSpNCHWOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> input(batch_size() * channels() * height() * width() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output(batch_size() * channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          float acc = 0.0f;
          for (size_t k = 0; k < height() * width(); k++) {
            acc += input[(i * channels() + j) * height() * width() + k];
          }
          output_ref[i * channels() + j] = acc / float(height() * width());
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
      ASSERT_EQ(xnn_status_success, xnn_initialize());
      xnn_operator_t global_average_pooling_op = nullptr;

      xnn_status status = xnn_create_global_average_pooling_spnchw_f32(
        channels(), output_min, output_max,
        0, &global_average_pooling_op);
      if (status == xnn_status_unsupported_parameter) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);

      // Smart pointer to automatically delete global_average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_average_pooling_op(global_average_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_average_pooling_spnchw_f32(
          global_average_pooling_op,
          batch_size(), height(), width(),
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
  size_t height_{1};
  size_t width_{1};
  size_t channels_{1};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
