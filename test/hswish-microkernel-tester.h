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


class HSwishMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline HSwishMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline HSwishMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline HSwishMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_hswish_ukernel_function hswish, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::fill(y.begin(), y.end(), std::nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Prepare micro-kernel parameters.
      union xnn_f32_hswish_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_hswish_params();
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_hswish_params();
          break;
      }

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = x_data[i] * std::max(std::min(x_data[i] + 3.0f, 6.0f), 0.0f) / 6.0f;
      }

      // Call optimized micro-kernel.
      hswish(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y_ref[i], y[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at position " << i << ", batch_size = " << batch_size();
      }
    }
  }

 private:
  size_t batch_size_{1};
  bool inplace_{false};
  size_t iterations_{5};
};
