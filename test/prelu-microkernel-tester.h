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
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


class PReLUMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline PReLUMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  inline size_t rows() const {
    return this->rows_;
  }

  inline PReLUMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline PReLUMicrokernelTester& input_stride(size_t input_stride) {
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

  inline PReLUMicrokernelTester& output_stride(size_t output_stride) {
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

  inline PReLUMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline PReLUMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline PReLUMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline PReLUMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_prelu_ukernel_function prelu, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32irng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);
    auto f32wrng = std::bind(std::uniform_real_distribution<float>(0.25f, 0.75f), rng);

    std::vector<float> x(channels() + (rows() - 1) * input_stride() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float, AlignedAllocator<float, 64>> w(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(channels() + (rows() - 1) * output_stride() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y_ref(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32irng));
      std::generate(w.begin(), w.end(), std::ref(f32wrng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32irng));
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results, without clamping.
      for (size_t i = 0; i < channels(); i++) {
        y_ref[i] = std::signbit(x_data[i]) ? x_data[i] * w[i] : x_data[i];
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_min = accumulated_range == 0.0f ?
        -std::numeric_limits<float>::infinity() : accumulated_min + accumulated_range / 255.0f * float(qmin());
      const float y_max = accumulated_range == 0.0f ?
        +std::numeric_limits<float>::infinity() : accumulated_max - accumulated_range / 255.0f * float(255 - qmax());

      // Prepare output parameters.
      xnn_f32_output_params output_params = { };
      switch (variant) {
        case Variant::Native:
          output_params = xnn_init_f32_output_params(y_min, y_max);
          break;
        case Variant::Scalar:
          output_params = xnn_init_scalar_f32_output_params(y_min, y_max);
          break;
      }

      // Clamp reference results.
      for (float& value : y_ref) {
        value = std::min(std::max(value, y_min), y_max);
      }

      // Call optimized micro-kernel.
      prelu(rows(), channels() * sizeof(float),
        x_data, input_stride() * sizeof(float),
        w.data(),
        y.data(), output_stride() * sizeof(float),
        &output_params);

      // Verify results.
      for (size_t i = 0; i < channels(); i++) {
        ASSERT_LE(y[i], y_max)
          << "at " << i << ", channels = " << channels();
        ASSERT_GE(y[i], y_min)
          << "at " << i << ", channels = " << channels();
        ASSERT_NEAR(y[i], y_ref[i], 1.0e-6f * std::abs(y_ref[i]))
          << "at " << i << ", channels = " << channels();
      }
    }
  }

 private:
  size_t rows_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  bool inplace_{false};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
