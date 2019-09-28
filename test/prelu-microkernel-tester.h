// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>


class PReLUMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline PReLUMicrokernelTester& m(size_t m) {
    assert(m != 0);
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline PReLUMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline PReLUMicrokernelTester& x_stride(size_t x_stride) {
    assert(x_stride != 0);
    this->x_stride_ = x_stride;
    return *this;
  }

  inline size_t x_stride() const {
    if (this->x_stride_ == 0) {
      return n();
    } else {
      assert(this->x_stride_ >= n());
      return this->x_stride_;
    }
  }

  inline PReLUMicrokernelTester& y_stride(size_t y_stride) {
    assert(y_stride != 0);
    this->y_stride_ = y_stride;
    return *this;
  }

  inline size_t y_stride() const {
    if (this->y_stride_ == 0) {
      return n();
    } else {
      assert(this->y_stride_ >= n());
      return this->y_stride_;
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

    std::vector<float> x(n() + (m() - 1) * x_stride() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float, AlignedAllocator<float, 16>> w(n() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(n() + (m() - 1) * y_stride() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y_ref(n());
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
      for (size_t i = 0; i < n(); i++) {
        y_ref[i] = signbit(x_data[i]) ? x_data[i] * w[i] : x_data[i];
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
          output_params = xnn_compute_f32_output_params(y_min, y_max);
          break;
        case Variant::Scalar:
          output_params = xnn_compute_scalar_f32_output_params(y_min, y_max);
          break;
      }

      // Clamp reference results.
      for (float& value : y_ref) {
        value = std::min(std::max(value, y_min), y_max);
      }

      // Call optimized micro-kernel.
      prelu(m(), n() * sizeof(float),
        x_data, x_stride() * sizeof(float),
        w.data(),
        y.data(), y_stride() * sizeof(float),
        &output_params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(y[i], y_max)
          << "at " << i << ", n = " << n();
        ASSERT_GE(y[i], y_min)
          << "at " << i << ", n = " << n();
        ASSERT_NEAR(y[i], y_ref[i], 1.0e-6f * std::abs(y_ref[i]))
          << "at " << i << ", n = " << n();
      }
    }
  }

 private:
  size_t m_{1};
  size_t n_{1};
  size_t x_stride_{0};
  size_t y_stride_{0};
  bool inplace_{false};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
