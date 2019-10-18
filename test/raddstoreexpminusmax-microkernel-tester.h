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


class RAddStoreExpMinusMaxMicrokernelTester {
 public:
  inline RAddStoreExpMinusMaxMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline RAddStoreExpMinusMaxMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(xnn_f32_raddstoreexpminusmax_ukernel_function raddstoreexpminusmax) const {
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
      std::fill(y.begin(), y.end(), std::nanf(""));

      // Compute reference results.
      double sum_ref = 0.0f;
      const float x_max = *std::max_element(x.begin(), x.begin() + n());
      for (size_t i = 0; i < n(); i++) {
        const double y_ref_value = exp(double(x[i]) - double(x_max));
        y_ref[i] = y_ref_value;
        sum_ref += y_ref_value;
      }

      // Call optimized micro-kernel.
      float sum = std::nanf("");
      raddstoreexpminusmax(n() * sizeof(float), x.data(), y.data(), &sum, x_max);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
      ASSERT_NEAR(y_ref[i], double(y[i]), std::abs(y_ref[i]) * 1.0e-6)
        << "i = " << i << ", n = " << n() << ", x_max = " << x_max;
      }
      ASSERT_NEAR(sum_ref, double(sum), std::abs(sum_ref) * 1.0e-6)
        << "n = " << n() << ", x_max = " << x_max;
    }
  }

 private:
  size_t n_{1};
  size_t iterations_{15};
};
