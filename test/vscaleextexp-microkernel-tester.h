// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/microfnptr.h>


class VScaleExtExpMicrokernelTester {
 public:
  inline VScaleExtExpMicrokernelTester& elements(size_t elements) {
    assert(elements != 0);
    this->elements_ = elements;
    return *this;
  }

  inline size_t elements() const {
    return this->elements_;
  }

  inline VScaleExtExpMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_vscaleextexp_ukernel_fn vscaleextexp) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    // Choose such range that expf(x[i]) overflows, but double-precision exp doesn't overflow.
    auto f32rng = std::bind(std::uniform_real_distribution<float>(90.0f, 100.0f), rng);

    std::vector<float> x(elements() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(elements());
    std::vector<double> y_ref(elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));

      // Compute scale parameters.
      double sum = 0.0;
      for (size_t i = 0; i < elements(); i++) {
        sum += std::exp(double(x[i]));
      }
      int sum_exponent;
      const double sum_mantissa = std::frexp(sum, &sum_exponent);
      const float scale_mantissa = float(1.0 / sum_mantissa);
      const float scale_exponent = -float(sum_exponent);

      // Compute reference results.
      for (size_t i = 0; i < elements(); i++) {
        y_ref[i] = std::exp(double(x[i])) / sum;
      }

      // Call optimized micro-kernel.
      vscaleextexp(elements() * sizeof(float), x.data(), y.data(), scale_mantissa, scale_exponent);

      // Verify results.
      for (size_t i = 0; i < elements(); i++) {
        EXPECT_NEAR(y_ref[i], y[i], std::abs(y_ref[i]) * 1.0e-6)
          << "elements = " << elements() << ", scale:mantissa = " << scale_mantissa << ", scale:exponent = " << scale_exponent;
      }
    }
  }

 private:
  size_t elements_{1};
  size_t iterations_{15};
};
