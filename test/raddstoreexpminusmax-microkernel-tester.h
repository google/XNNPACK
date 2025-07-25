// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_TEST_RADDSTOREEXPMINUSMAX_MICROKERNEL_TESTER_H_
#define XNNPACK_TEST_RADDSTOREEXPMINUSMAX_MICROKERNEL_TESTER_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams.h"
#include "test/replicable_random_device.h"

class RAddStoreExpMinusMaxMicrokernelTester {
 public:
  RAddStoreExpMinusMaxMicrokernelTester& elements(size_t elements) {
    assert(elements != 0);
    this->elements_ = elements;
    return *this;
  }

  size_t elements() const { return this->elements_; }

  RAddStoreExpMinusMaxMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  void Test(xnn_f16_raddstoreexpminusmax_ukernel_fn raddstoreexpminusmax,
            xnn_init_f16_expminus_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    // Choose such range that exph(x[i]) overflows, but exph(x[i] - x_max)
    // doesn't. However, the range is still narrow enough that double-precision
    // exp doesn't overflow.
    std::uniform_real_distribution<float> f32dist(15.0f, 20.0f);

    xnnpack::Buffer<xnn_float16> x(elements(), xnnpack::XnnExtraBytes);
    xnnpack::Buffer<xnn_float16> y(elements());
    xnnpack::Buffer<float> y_ref(elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });

      // Compute reference results.
      float sum_ref = 0.0f;
      float x_max_as_float = -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < elements(); i++) {
        x_max_as_float = std::max<float>(x_max_as_float, x[i]);
      }
      const xnn_float16 x_max_as_half =
          static_cast<xnn_float16>(x_max_as_float);
      for (size_t i = 0; i < elements(); i++) {
        const float y_ref_value = exp(x[i] - x_max_as_float);
        y_ref[i] = y_ref_value;
        sum_ref += y_ref_value;
      }

      // Call optimized micro-kernel.
      xnn_float16 sum;
      raddstoreexpminusmax(elements() * sizeof(xnn_float16), x.data(),
                           &x_max_as_half, y.data(), &sum, nullptr);

      // Verify results.
      for (size_t i = 0; i < elements(); i++) {
        ASSERT_NEAR(y_ref[i], y[i], std::abs(y_ref[i]) * 5.0e-3f)
            << "element " << i << " / " << elements() << ", x_max "
            << x_max_as_float;
      }
      ASSERT_NEAR(sum_ref, sum, std::abs(sum_ref) * 5.0e-3f)
          << "batch " << elements() << ", x_max " << x_max_as_float;
    }
  }

  void Test(xnn_f32_raddstoreexpminusmax_ukernel_fn raddstoreexpminusmax,
            xnn_init_f32_expminus_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    // Choose such range that expf(x[i]) overflows, but expf(x[i] - x_max)
    // doesn't. However, the range is still narrow enough that double-precision
    // exp doesn't overflow.
    std::uniform_real_distribution<float> f32dist(90.0f, 100.0f);

    xnnpack::Buffer<float> x(elements(), xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> y(elements());
    xnnpack::Buffer<double> y_ref(elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });

      // Compute reference results.
      double sum_ref = 0.0f;
      const float x_max = *std::max_element(x.begin(), x.begin() + elements());
      for (size_t i = 0; i < elements(); i++) {
        const double y_ref_value = exp(double(x[i]) - double(x_max));
        y_ref[i] = y_ref_value;
        sum_ref += y_ref_value;
      }

      // Call optimized micro-kernel.
      float sum;
      raddstoreexpminusmax(elements() * sizeof(float), x.data(), &x_max,
                           y.data(), &sum, nullptr);

      // Verify results.
      for (size_t i = 0; i < elements(); i++) {
        ASSERT_NEAR(y_ref[i], double(y[i]), std::abs(y_ref[i]) * 1.0e-6)
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

#endif  // XNNPACK_TEST_RADDSTOREEXPMINUSMAX_MICROKERNEL_TESTER_H_
