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
#include <xnnpack/params.h>


class VScaleExpMinusMaxMicrokernelTester {
 public:
  inline VScaleExpMinusMaxMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline VScaleExpMinusMaxMicrokernelTester& scale(float scale) {
    assert(std::isfinite(scale));
    assert(scale > 0);
    this->scale_ = scale;
    return *this;
  }

  inline size_t scale() const {
    return this->scale_;
  }

  inline VScaleExpMinusMaxMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(xnn_f32_vscaleexpminusmax_ukernel_function vscaleexpminusmax) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    // Choose such range that expf(x[i]) overflows, but expf(x[i] - x_max) doesn't.
    // However, the range is still narrow enough that double-precision exp doesn't overflow.
    auto f32rng = std::bind(std::uniform_real_distribution<float>(90.0f, 100.0f), rng);

    std::vector<float> x(n() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(n());
    std::vector<double> y_ref(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));

      // Compute reference results.
      const float x_max = *std::max_element(x.begin(), x.begin() + n());
      for (size_t i = 0; i < n(); i++) {
        y_ref[i] = double(scale()) * exp(double(x[i]) - double(x_max));
      }

      // Call optimized micro-kernel.
      vscaleexpminusmax(n() * sizeof(float), x.data(), y.data(), scale(), x_max);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        ASSERT_NEAR(y_ref[i], y[i], std::abs(y_ref[i]) * 1.0e-6)
          << "n = " << n() << ", scale = " << scale() << ", x_max = " << x_max;
      }
    }
  }

 private:
  size_t n_{1};
  float scale_{1.0f};
  size_t iterations_{15};
};
