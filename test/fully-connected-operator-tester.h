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


class FullyConnectedOperatorTester {
 public:
  inline FullyConnectedOperatorTester& input_channels(size_t input_channels) {
    assert(input_channels >= 1);
    this->input_channels_ = input_channels;
    return *this;
  }

  inline size_t input_channels() const {
    return this->input_channels_;
  }

  inline FullyConnectedOperatorTester& output_channels(size_t output_channels) {
    assert(output_channels >= 1);
    this->output_channels_ = output_channels;
    return *this;
  }

  inline size_t output_channels() const {
    return this->output_channels_;
  }

  inline FullyConnectedOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size >= 1);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline FullyConnectedOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride >= 1);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return input_channels();
    } else {
      assert(this->input_stride_ >= input_channels());
      return this->input_stride_;
    }
  }

  inline FullyConnectedOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride >= 1);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return output_channels();
    } else {
      assert(this->output_stride_ >= output_channels());
      return this->output_stride_;
    }
  }

  inline FullyConnectedOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline FullyConnectedOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline FullyConnectedOperatorTester& transpose_weights(bool transpose_weights) {
    this->transpose_weights_ = transpose_weights;
    return *this;
  }

  inline bool transpose_weights() const {
    return this->transpose_weights_;
  }

  inline FullyConnectedOperatorTester& has_bias(bool has_bias) {
    this->has_bias_ = has_bias;
    return *this;
  }

  inline bool has_bias() const {
    return this->has_bias_;
  }

  inline FullyConnectedOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestQ8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<uint8_t> kernel(output_channels() * input_channels());
    std::vector<int32_t> bias(output_channels());
    std::vector<uint8_t> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<int32_t> accumulators(batch_size() * output_channels());
    std::vector<double> output_ref(batch_size() * output_channels());

    const uint8_t input_zero_point = 127;
    const uint8_t kernel_zero_point = 127;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            accumulators[i * output_channels() + oc] = bias[oc];
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      if (transpose_weights()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              accumulators[i * output_channels() + oc] +=
                (int32_t(input[i * input_stride() + ic]) - int32_t(input_zero_point)) *
                (int32_t(kernel[ic * output_channels() + oc]) - int32_t(kernel_zero_point));
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              accumulators[i * output_channels() + oc] +=
                (int32_t(input[i * input_stride() + ic]) - int32_t(input_zero_point)) *
                (int32_t(kernel[oc * input_channels() + ic]) - int32_t(kernel_zero_point));
            }
          }
        }
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
      const uint8_t output_zero_point = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      // Renormalize reference results.
      std::transform(accumulators.cbegin(), accumulators.cend(), output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax()) - output_zero_point), double(qmin()) - output_zero_point);
        });

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t fully_connected_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_fully_connected_nc_q8(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          input_zero_point, 1.0f /* input scale */,
          kernel_zero_point, 1.0f /* kernel scale */,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, qmin(), qmax(),
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          &fully_connected_op));

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_q8(
          fully_connected_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < output_channels(); c++) {
          ASSERT_LE(int32_t(output[i * output_stride() + c]), int32_t(qmax()))
            << "batch index = " << i << ", channel = " << c;
          ASSERT_GE(int32_t(output[i * output_stride() + c]), int32_t(qmin()))
            << "batch index = " << i << ", channel = " << c;
          ASSERT_NEAR(
              output_ref[i * output_channels() + c],
              double(output[i * output_stride() + c]) - double(output_zero_point),
              0.9)
            << "batch index = " << i << ", channel = " << c;
        }
      }
    }
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.1f, 1.0f), rng);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<float> kernel(output_channels() * input_channels());
    std::vector<float> bias(output_channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            output_ref[i * output_channels() + oc] = bias[oc];
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      if (transpose_weights()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                input[i * input_stride() + ic] * kernel[ic * output_channels() + oc];
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                input[i * input_stride() + ic] * kernel[oc * input_channels() + ic];
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = qmin() == 0 ? -std::numeric_limits<float>::infinity() :
        accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = qmax() == 255 ? std::numeric_limits<float>::infinity() :
        accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t fully_connected_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_fully_connected_nc_f32(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          &fully_connected_op));

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_f32(
          fully_connected_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < output_channels(); c++) {
          ASSERT_LE(output[i * output_stride() + c], output_max)
            << "batch index = " << i << ", channel = " << c;
          ASSERT_GE(output[i * output_stride() + c], output_min)
            << "batch index = " << i << ", channel = " << c;
          ASSERT_NEAR(
              output_ref[i * output_channels() + c],
              output[i * output_stride() + c],
              1.0e-4 * std::abs(output_ref[i * output_channels() + c]))
            << "batch index = " << i << ", channel = " << c;
        }
      }
    }
  }

 private:
  size_t input_channels_{1};
  size_t input_stride_{0};
  size_t output_channels_{1};
  size_t output_stride_{0};
  size_t batch_size_{1};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  bool transpose_weights_{false};
  bool has_bias_{true};
  size_t iterations_{1};
};
