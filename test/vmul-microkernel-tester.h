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
#include <xnnpack/params.h>
#include <xnnpack/params-init.h>


class VMulMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline VMulMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline VMulMicrokernelTester& inplace_a(bool inplace_a) {
    this->inplace_a_ = inplace_a;
    return *this;
  }

  inline bool inplace_a() const {
    return this->inplace_a_;
  }

  inline VMulMicrokernelTester& inplace_b(bool inplace_b) {
    this->inplace_b_ = inplace_b;
    return *this;
  }

  inline bool inplace_b() const {
    return this->inplace_b_;
  }

  inline VMulMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline VMulMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline VMulMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_vmul_ukernel_function vmul, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    std::vector<float> a(n() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(n() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(n() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      if (inplace_a() || inplace_b()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* a_data = inplace_a() ? y.data() : a.data();
      const float* b_data = inplace_b() ? y.data() : b.data();

      // Compute reference results.
      for (size_t i = 0; i < n(); i++) {
        y_ref[i] = a_data[i] * b_data[i];
      }
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      if (n() > 1) {
        ASSERT_GT(accumulated_range, 0.0f) << "n = " << n();
      }
      const float y_max = accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      const float y_min = accumulated_min + accumulated_range / 255.0f * float(qmin());
      for (size_t i = 0; i < n(); i++) {
        y_ref[i] = std::max<float>(std::min<float>(y_ref[i], y_max), y_min);
      }

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

      // Call optimized micro-kernel.
      vmul(n() * sizeof(float), a_data, b_data, y.data(), &output_params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        ASSERT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at " << i << ", n = " << n();
      }
    }
  }

 private:
  size_t n_{1};
  bool inplace_a_{false};
  bool inplace_b_{false};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
