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
  inline VScaleExpMinusMaxMicrokernelTester& elements(size_t elements) {
    assert(elements != 0);
    this->elements_ = elements;
    return *this;
  }

  inline size_t elements() const {
    return this->elements_;
  }

  inline VScaleExpMinusMaxMicrokernelTester& scale(float scale) {
    assert(std::isfinite(scale));
    assert(scale > 0);
    this->scale_ = scale;
    return *this;
  }

  inline float scale() const {
    return this->scale_;
  }

  inline VScaleExpMinusMaxMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_vscaleexpminusmax_ukernel_function vscaleexpminusmax) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    // Choose such range that expf(x[i]) overflows, but expf(x[i] - x_max) doesn't.
    // However, the range is still narrow enough that double-precision exp doesn't overflow.
    auto f32rng = std::bind(std::uniform_real_distribution<float>(90.0f, 100.0f), rng);

    std::vector<float> x(elements() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(elements());
    std::vector<double> y_ref(elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));

      // Compute reference results.
      const float x_max = *std::max_element(x.begin(), x.begin() + elements());
      for (size_t i = 0; i < elements(); i++) {
        y_ref[i] = double(scale()) * exp(double(x[i]) - double(x_max));
      }

      // Call optimized micro-kernel.
      vscaleexpminusmax(elements() * sizeof(float), x.data(), y.data(), scale(), x_max);

      // Verify results.
      for (size_t i = 0; i < elements(); i++) {
        ASSERT_NEAR(y_ref[i], y[i], std::abs(y_ref[i]) * 1.0e-6)
          << "elements = " << elements() << ", scale = " << scale() << ", x_max = " << x_max;
      }
    }
  }

 private:
  size_t elements_{1};
  float scale_{1.0f};
  size_t iterations_{15};
};
