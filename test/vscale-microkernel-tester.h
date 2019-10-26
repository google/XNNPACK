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
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


class VScaleMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline VScaleMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline VScaleMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline VScaleMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_vscale_ukernel_function vscale, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);

    std::vector<float> x(n() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(n() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float c = f32rng();
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < n(); i++) {
        y_ref[i] = x_data[i] * c;
      }

      // Call optimized micro-kernel.
      vscale(n() * sizeof(float), x_data, y.data(), c);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        ASSERT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at " << i << ", n = " << n();
      }
    }
  }

 private:
  size_t n_{1};
  size_t iterations_{15};
  bool inplace_{false};
};
