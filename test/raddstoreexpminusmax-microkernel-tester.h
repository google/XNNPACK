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
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


class RAddStoreExpMinusMaxMicrokernelTester {
 public:
  inline RAddStoreExpMinusMaxMicrokernelTester& elements(size_t elements) {
    assert(elements != 0);
    this->elements_ = elements;
    return *this;
  }

  inline size_t elements() const {
    return this->elements_;
  }

  inline RAddStoreExpMinusMaxMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_raddstoreexpminusmax_ukernel_fn raddstoreexpminusmax, xnn_init_f16_expminus_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    // Choose such range that exph(x[i]) overflows, but exph(x[i] - x_max) doesn't.
    // However, the range is still narrow enough that double-precision exp doesn't overflow.
    std::uniform_real_distribution<float> f32dist(15.0f, 20.0f);

    std::vector<uint16_t> x(elements() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> y(elements());
    std::vector<float> y_ref(elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results.
      float sum_ref = 0.0f;
      float x_max_as_float = -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < elements(); i++) {
        x_max_as_float = std::max(x_max_as_float, fp16_ieee_to_fp32_value(x[i]));
      }
      const uint16_t x_max_as_half = fp16_ieee_from_fp32_value(x_max_as_float);
      for (size_t i = 0; i < elements(); i++) {
        const float y_ref_value = exp(fp16_ieee_to_fp32_value(x[i]) - x_max_as_float);
        y_ref[i] = y_ref_value;
        sum_ref += y_ref_value;
      }

      // Call optimized micro-kernel.
      uint16_t sum = UINT16_C(0x7E00) /* NaN */;
      xnn_f16_expminus_params params;
      init_params(&params);
      raddstoreexpminusmax(elements() * sizeof(uint16_t), x.data(), &x_max_as_half, y.data(), &sum, &params);

      // Verify results.
      for (size_t i = 0; i < elements(); i++) {
      EXPECT_NEAR(y_ref[i], fp16_ieee_to_fp32_value(y[i]), std::abs(y_ref[i]) * 5.0e-3f)
        << "element " << i << " / " << elements() << ", x_max " << x_max_as_float;
      }
      ASSERT_NEAR(sum_ref, fp16_ieee_to_fp32_value(sum), std::abs(sum_ref) * 5.0e-3f)
        << "batch " << elements() << ", x_max " << x_max_as_float;
    }
  }

  void Test(xnn_f32_raddstoreexpminusmax_ukernel_fn raddstoreexpminusmax, xnn_init_f32_expminus_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    // Choose such range that expf(x[i]) overflows, but expf(x[i] - x_max) doesn't.
    // However, the range is still narrow enough that double-precision exp doesn't overflow.
    std::uniform_real_distribution<float> f32dist(90.0f, 100.0f);

    std::vector<float> x(elements() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(elements());
    std::vector<double> y_ref(elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      std::fill(y.begin(), y.end(), std::nanf(""));

      // Compute reference results.
      double sum_ref = 0.0f;
      const float x_max = *std::max_element(x.begin(), x.begin() + elements());
      for (size_t i = 0; i < elements(); i++) {
        const double y_ref_value = exp(double(x[i]) - double(x_max));
        y_ref[i] = y_ref_value;
        sum_ref += y_ref_value;
      }

      // Call optimized micro-kernel.
      float sum = std::nanf("");
      xnn_f32_expminus_params params;
      init_params(&params);
      raddstoreexpminusmax(elements() * sizeof(float), x.data(), &x_max, y.data(), &sum, &params);

      // Verify results.
      for (size_t i = 0; i < elements(); i++) {
      EXPECT_NEAR(y_ref[i], double(y[i]), std::abs(y_ref[i]) * 1.0e-6)
        << "element " << i << " / " << elements() << ", x_max " << x_max;
      }
      ASSERT_NEAR(sum_ref, double(sum), std::abs(sum_ref) * 1.0e-6)
        << "batch " << elements() << ", x_max " << x_max;
    }
  }

 private:
  size_t elements_{1};
  size_t iterations_{15};
};
