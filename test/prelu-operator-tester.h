// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

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

  inline PReLUOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF32() const {
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
          ASSERT_NEAR(y[i * y_stride() + c], y_ref[i * channels() + c], 1.0e-6f * std::abs(y_ref[i * channels() + c]))
            << "i = " << i << ", c = " << c;
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t channels_{1};
  size_t x_stride_{0};
  size_t y_stride_{0};
  size_t iterations_{15};
};
