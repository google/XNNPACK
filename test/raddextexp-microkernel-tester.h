// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

class RAddExtExpMicrokernelTester {
 public:
  RAddExtExpMicrokernelTester& elements(size_t elements) {
    assert(elements != 0);
    this->elements_ = elements;
    return *this;
  }

  size_t elements() const {
    return this->elements_;
  }

  RAddExtExpMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_raddextexp_ukernel_fn raddextexp) const {
    xnnpack::ReplicableRandomDevice rng;
    // Choose such range that expf(x[i]) overflows, but double-precision exp doesn't overflow.
    auto f32rng = [&rng]() {
      return std::uniform_real_distribution<float>(90.0f, 100.0f)(rng);
    };

    std::vector<float> x(elements() + XNN_EXTRA_BYTES / sizeof(float));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));

      // Compute reference results.
      double sum_ref = 0.0f;
      for (size_t i = 0; i < elements(); i++) {
        sum_ref += exp(double(x[i]));
      }

      // Call optimized micro-kernel.
      float sum[2] = { nanf(""), nanf("") };
      raddextexp(elements() * sizeof(float), x.data(), sum);

      // Verify results.
      ASSERT_NEAR(sum_ref, exp2(double(sum[1])) * double(sum[0]), std::abs(sum_ref) * 1.0e-6)
        << "elements = " << elements() << ", y:value = " << sum[0] << ", y:exponent = " << sum[1];
    }
  }

 private:
  size_t elements_{1};
  size_t iterations_{15};
};
