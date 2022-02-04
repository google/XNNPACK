// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>


class PReLUOperatorTester {
 public:
  enum class WeightsType {
    Default,
    FP32,
  };

  inline PReLUOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline PReLUOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline PReLUOperatorTester& x_stride(size_t x_stride) {
    assert(x_stride != 0);
    this->x_stride_ = x_stride;
    return *this;
  }

  inline size_t x_stride() const {
    if (this->x_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->x_stride_ >= this->channels_);
      return this->x_stride_;
    }
  }

  inline PReLUOperatorTester& y_stride(size_t y_stride) {
    assert(y_stride != 0);
    this->y_stride_ = y_stride;
    return *this;
  }

  inline size_t y_stride() const {
    if (this->y_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->y_stride_ >= this->channels_);
      return this->y_stride_;
    }
  }

  inline PReLUOperatorTester& weights_type(WeightsType weights_type) {
    this->weights_type_ = weights_type;
    return *this;
  }

  inline WeightsType weights_type() const {
    return this->weights_type_;
  }

  inline PReLUOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF16() const {
    switch (weights_type()) {
      case WeightsType::Default:
        break;
      case WeightsType::FP32:
        break;
      default:
        GTEST_FAIL() << "unexpected weights type";
    }

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32irng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);
    auto f16irng = std::bind(fp16_ieee_from_fp32_value, f32irng);
    auto f32wrng = std::bind(std::uniform_real_distribution<float>(0.25f, 0.75f), rng);
    auto f16wrng = std::bind(fp16_ieee_from_fp32_value, f32wrng);

    std::vector<uint16_t> x((batch_size() - 1) * x_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> w(channels());
    std::vector<float> w_as_float(channels());
    std::vector<uint16_t> y((batch_size() - 1) * y_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<float> y_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f16irng));
      std::generate(w.begin(), w.end(), std::ref(f16wrng));
      std::transform(w.cbegin(), w.cend(), w_as_float.begin(), fp16_ieee_to_fp32_value);
      std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x_value = fp16_ieee_to_fp32_value(x[i * x_stride() + c]);
          const float w_value = w_as_float[c];
          y_ref[i * channels() + c] = std::signbit(x_value) ? x_value * w_value : x_value;
        }
      }

      // Create, setup, run, and destroy PReLU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t prelu_op = nullptr;

      const void* negative_slope_data = w.data();
      if (weights_type() == WeightsType::FP32) {
        negative_slope_data = w_as_float.data();
      }
      uint32_t flags = 0;
      if (weights_type() == WeightsType::FP32) {
        flags |= XNN_FLAG_FP32_STATIC_WEIGHTS;
      }
      ASSERT_EQ(xnn_status_success,
        xnn_create_prelu_nc_f16(
          channels(), x_stride(), y_stride(),
          negative_slope_data,
          flags, &prelu_op));
      ASSERT_NE(nullptr, prelu_op);

      // Smart pointer to automatically delete prelu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_prelu_op(prelu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_prelu_nc_f16(
          prelu_op,
          batch_size(),
          x.data(), y.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(prelu_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              fp16_ieee_to_fp32_value(y[i * y_stride() + c]),
              y_ref[i * channels() + c],
              std::max(1.0e-4f, std::abs(y_ref[i * channels() + c]) * 1.0e-4f))
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestF32() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32irng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);
    auto f32wrng = std::bind(std::uniform_real_distribution<float>(0.25f, 0.75f), rng);

    std::vector<float> x((batch_size() - 1) * x_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> w(channels());
    std::vector<float> y((batch_size() - 1) * y_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32irng));
      std::generate(w.begin(), w.end(), std::ref(f32wrng));
      std::fill(y.begin(), y.end(), nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          y_ref[i * channels() + c] = std::signbit(x[i * x_stride() + c]) ? x[i * x_stride() + c] * w[c] : x[i * x_stride() + c];
        }
      }

      // Create, setup, run, and destroy PReLU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t prelu_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_prelu_nc_f32(
          channels(), x_stride(), y_stride(),
          w.data(),
          0, &prelu_op));
      ASSERT_NE(nullptr, prelu_op);

      // Smart pointer to automatically delete prelu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_prelu_op(prelu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_prelu_nc_f32(
          prelu_op,
          batch_size(),
          x.data(), y.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(prelu_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              y[i * y_stride() + c],
              y_ref[i * channels() + c],
              std::max(1.0e-6f, std::abs(y_ref[i * channels() + c]) * 1.0e-6f))
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t channels_{1};
  size_t x_stride_{0};
  size_t y_stride_{0};
  WeightsType weights_type_{WeightsType::Default};
  size_t iterations_{15};
};
